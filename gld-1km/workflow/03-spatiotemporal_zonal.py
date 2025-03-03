import sys
sys.path.insert(0,'./scikit-map')
import skmap
print(skmap.__file__)


from pathlib import Path
from rasterio.windows import from_bounds
from rasterio.windows import Window
from skmap import io
from skmap.misc import make_tempdir
from skmap.misc import ttprint
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio 
import re 
import skmap_bindings as skb

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Table

import traceback
import bottleneck as bn

from rasterio.features import rasterize
import rasterio

def _covs(layernames, prefixes, n_prefixes):
    return layernames[
        np.logical_and(
            np.logical_or.reduce(
                [ layernames.str.contains(p) for p in prefixes ]
            ),
            np.logical_not(np.logical_or.reduce(
                [ layernames.str.contains(p) for p in n_prefixes ]
        ))
    )]

def _raster_layer(static_covs, temporal_covs, years, base_path):
    
    path, rtype = [], []
    if static_covs is not None:
        path += [ f'{base_path}{c}.vrt' for c in static_covs ]
        rtype += [ 'static' for c in static_covs ]
    
    for y in years:
        path += [ f"{base_path}{c.replace('{year}', str(y))}.vrt" for c in temporal_covs ]
        rtype += [ y for c in temporal_covs ]
    
    return pd.DataFrame({
        'path': path,
        'type': rtype
    })

def _mask_layers(years, base_path):
    
    temporal_covs = [
        'gpw_livestock.systems_grassland.cropland.rules_c_1km_{year}0101_{year}1231_go_epsg.4326_v1',
        'gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{year}0101_{year}1231_go_epsg.4326_v1'
    ]
    
    return _raster_layer(None, temporal_covs, years, base_path)

def _years(polygon_samp):
    cols = polygon_samp.columns[polygon_samp.columns.str.contains('_2')]
    cols = cols[(polygon_samp[cols] >= 0).to_numpy().flatten()]
    years = cols.map(lambda x: int(x.split('_')[1])).unique()
    years = years.drop(2022, errors='ignore')
    years = years.drop(2023, errors='ignore')
    
    return sorted(years)

def _window(polygon_samp, df_layer, crs):
    raster_layers = list(df_layer['path'])
    #crs_igg = '+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
    minx, miny, maxx, maxy = polygon_samp.to_crs(crs).total_bounds
    return  from_bounds(minx, miny, maxx, maxy, rasterio.open(raster_layers[0]).transform).round_lengths().round_offsets()

wd = '/mnt/tupi/WRI/livestock_global_modeling/'

#polygons_fn = f'{wd}livestock_census_raw/gpw_livestock.animals_gpw.fao.faostat.malek.2024_polygon.samples_20000101_20231231_go_epsg.4326_v1.gpkg'
#polygon_samples = gpd.read_file(polygons_fn)

polygon_samples = joblib.load(f'{wd}livestock_census_ard/gpw_livestock.animals_gpw.fao.glw3_polygon.samples_20000101_20231231_go_epsg.4326_v1.lz4')
livestock_covs = pd.read_csv(f'{wd}/livestock_census_ard/livestock_cov.csv')

prefixes = [
    'bare', 'crop', 'clm_accum.precipitation', 'clm_lst_mod11a2.nighttime_p50', 'clm_lst_mod11a2.nighttime.trend_p50',
    'clm_lst_mod11a2.daytime_p50', 'clm_lst_mod11a2.daytime.trend_p50', 'clm_lst_max.geom', 'clm_lst_min.geom', 
    'veg_blue_mod13q1.v061_p50_', 'veg_mir_mod13q1.v061_p50_', 'veg_red_mod13q1.v061_p50_', 'veg_nir_mod13q1.v061_p50_',  'veg_ndvi_mod13q1.v061.trend_p50_', 
    'easterness_','filtered.dtm_','flow.accum_','geomorphon_', 'hillshade_','lcv_', 'neg.openness_', 'northerness_', 'nosink_', 
    'pop.count_','pos.openness_', 'slope_','spec.catch.area.factor_','wilderness_','wv_','surface.water','wetlands_',
    'lcv_wetlands','forest.cover', 'gdp.per.capita', 'hdi', 'religion','rural.pop.dist'
]
n_prefixes = ['clm_lst_mod11a2.nighttime_sd', 'clm_lst_mod11a2.daytime_sd', 
              'lcv_water.distance_glad.interanual.dynamic.classes', 'lcv_bare.surface_landsat.', 'bsf']

static_covs = _covs(livestock_covs[livestock_covs['type'] == 'static'].layername, prefixes, n_prefixes)
temporal_covs = _covs(livestock_covs[livestock_covs['type'] == 'temporal'].layername, prefixes, n_prefixes)

out_file = 'livestock_zonal_ultimate.pq'

n_jobs = 96

processed_index = None

try:
  ttprint(f"Reading file {out_file}")
  processed_index = pd.read_parquet(out_file)['polygon_idx']
  ttprint(f"End.")
except:
  pass

start_i = int(sys.argv[1])
end_i = int(sys.argv[2])

for index, _ in polygon_samples[start_i:end_i].iterrows():

  if processed_index is not None:
    if np.sum(processed_index == index).astype('int') > 0:
      ttprint(f"Polygon {index} already processed.")
      continue

  polygon_samp = polygon_samples[polygon_samples.index == index]
  ttprint(f"Zonal hist for polygon {index}")
  
  try:
    base_path = '/mnt/tupi/WRI/livestock_global_modeling/vrt/'
    crs_igg = '+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

    years = _years(polygon_samp) 
    #years = range(2000, 2021)
    
    df_layer = _raster_layer(static_covs, temporal_covs, years, base_path)
    df_mask = _mask_layers(years, base_path)
    window = _window(polygon_samp, df_layer, crs_igg)

    covs_data = io.read_rasters_cpp(df_layer['path'], window=window, verbose=False, n_jobs=96)
    mask_data = io.read_rasters_cpp(df_mask['path'], window=window, verbose=False, n_jobs=96)
    
    ds = rasterio.open(df_mask['path'][0])
    transform = (
      ds.transform[0], 0.0,  ds.transform[2] + window.col_off * ds.transform[0],
      0.0, ds.transform[4],  ds.transform[5] + window.row_off * ds.transform[4]
    )
    out_shape = (window.height,window.width)
    polygon_mask_temp = rasterize(polygon_samp.to_crs(crs_igg).geometry, fill=0, out_shape=out_shape, transform=transform, dtype='float32')
    polygon_mask = np.empty((mask_data.shape[1],), dtype='float32')
    polygon_mask[:] = polygon_mask_temp.flatten()
    print(f'Fixed version using {out_shape}')

    df_livestock_sys = df_mask[df_mask.path.str.contains('gpw_livestock.systems_grassland.cropland.rules')]
    masked_zonal = {}

    for year in years:
        
      df_layer_y = df_layer[np.logical_or.reduce([
          df_layer['type'] == year,
          df_layer['type'] == 'static',
      ])].copy()

      df_layer_y.loc[:,'label'] = df_layer_y['path'].apply(lambda f: Path(f).stem)
      df_layer_y.loc[df_layer_y['type'] != 'static','label'] = df_layer_y[df_layer_y['type'] != 'static']['label'].apply(lambda f: f.replace(f'_{year}','_year').replace(f'..{year}','..year'))

      covs_idx = list(df_layer_y.index)
      covs_label = list(df_layer_y['label'])
      covs_label += [ 'livestock_area_km' ]

      for idx, row_mask in df_livestock_sys[df_livestock_sys['type'] == year].iterrows():
          
        mask_layer = Path(row_mask['path']).stem
        
        val = 0
        key = f'{mask_layer}_gt.{int(val)}'

        n_covs = len(covs_idx)
        mask = np.logical_and((mask_data[idx,:] > val), (polygon_mask == 1)).astype('float32')
        covs_data_idx = np.empty((n_covs, covs_data.shape[1]), dtype='float32')
        covs_data_idx[:] = covs_data[covs_idx,:]
        skb.maskDataRows(covs_data_idx, n_jobs, range(0,n_covs), mask, 0, np.nan)
        mean_data = np.empty((len(covs_idx)), dtype='float32')
        skb.nanMean(covs_data_idx, n_jobs, mean_data)
        mean_data = np.round(mean_data)

        land_data = np.empty((1, covs_data.shape[1]), dtype='float32')
        land_data[:] = mask_data[idx + 1,:] # Accessing gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{year}0101_{year}1231_go_epsg.4326_v1
        skb.maskDataRows(land_data, n_jobs, [0], mask, 0, np.nan)
        sum_land = bn.nansum(land_data) / 100
        mean_data = np.append(mean_data, sum_land)

        masked_zonal[key] = np.round(mean_data)
                
    ttprint(f"End")

    zonal_df = pd.DataFrame(masked_zonal.values(), columns=covs_label)
    zonal_df['mask_layer'] = masked_zonal.keys()
    zonal_df['polygon_idx'] = index #polygon_samp.index[0]
    #zonal_df['livestock_system'] = zonal_df['mask_layer'].str.split('eq.', expand=True)[1]
    zonal_df['year'] = pd.to_numeric(zonal_df['mask_layer'].str.split('_', expand=True)[5].str[0:4])
      
    pq.write_to_dataset(
      Table.from_pandas(zonal_df),
      out_file,
      partition_cols=['polygon_idx'],
      compression="snappy",
      version="2.4",
    )

  except:
    print(traceback.format_exc())
    ttprint(f"Error in polygon {index}")
  #break