import sys
sys.path.insert(0,'./scikit-map')
import skmap
print(skmap.__file__)

from pathlib import Path
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from skmap.misc import ttprint
from skmap import io

import bottleneck as bn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skmap_bindings as skb
import numpy as np
import math

import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Table

import traceback

def extract_years(polygon_samp):
    cols = polygon_samp.columns[polygon_samp.columns.str.contains('_2')]
    cols = cols[(polygon_samp[cols] >= 0).to_numpy().flatten()]
    years = cols.map(lambda x: int(x.split('_')[1])).unique()
    #years = years.drop(2023, errors='ignore')
    #years = years.drop(2024, errors='ignore')
    return sorted(years)

def extract_window(polygon_samp, datacube_df, crs):
    raster_layers = list(datacube_df['path'])
    minx, miny, maxx, maxy = polygon_samp.to_crs(crs).total_bounds
    return  from_bounds(minx, miny, maxx, maxy, rasterio.open(raster_layers[0]).transform).round_lengths().round_offsets()

def gen_covs_info(static_covs, temporal_covs, years, base_path):
    
    path, rtype = [], []
    if static_covs is not None:
        path += [ f'{base_path}/{c}.vrt' for c in static_covs ]
        rtype += [ 'static' for c in static_covs ]
    
    for y in years:
        path += [ f"{base_path}/{c.replace('{year}', str(y))}.vrt" for c in temporal_covs ]
        rtype += [ y for c in temporal_covs ]
    
    return pd.DataFrame({
        'path': path,
        'type': rtype
    })

def gen_mask_info(years, base_path):
    
    temporal_covs = [
        'gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{year}0101_{year}1231_go_epsg.4326_v1'
    ]
    
    return gen_covs_info(None, temporal_covs, years, base_path)

def gen_polygon_mask(polygon_samp, mask_info, crs):
    window = extract_window(polygon_samp, mask_info, crs)
    ds = rasterio.open(mask_info['path'][0])
    transform = (
      ds.transform[0], 0.0,  ds.transform[2] + window.col_off * ds.transform[0],
      0.0, ds.transform[4],  ds.transform[5] + window.row_off * ds.transform[4]
    )
    out_shape = (window.height,window.width)
    polygon_mask_temp = rasterize(polygon_samp.to_crs(crs).geometry, fill=0, out_shape=out_shape, transform=transform, dtype='float32')
    polygon_mask = np.empty((window.height * window.width,), dtype='float32')
    polygon_mask[:] = polygon_mask_temp.flatten()
    return polygon_mask

def spationtemporal_masked_zonal_mean(polygon_mask, covs_data, covs_info, mask_data, mask_info, mask_ignore_val, n_jobs = 4, verbose=False):
    masked_zonal = {}

    for year in mask_info['type']:

        covs_info_y = covs_info[np.logical_or.reduce([
          covs_info['type'] == year,
          covs_info['type'] == 'static',
        ])].copy()

        covs_info_y.loc[:,'label'] = covs_info_y['path'].apply(lambda f: Path(f).stem)
        covs_info_y.loc[covs_info_y['type'] != 'static','label'] = covs_info_y[covs_info_y['type'] != 'static']['label'].apply(lambda f: f.replace(f'_{year}','_year').replace(f'..{year}','..year'))

        covs_idx = list(covs_info_y.index)
        covs_label = list(covs_info_y['label'])
        covs_label += [ 'mask_sum' ]

        row_mask = mask_info[mask_info['type'] == year].iloc[0]
        idx = row_mask.name
        key = year
        n_covs = len(covs_idx)

        mask = np.logical_and(
            (mask_data[idx,:] != mask_ignore_val), 
            (polygon_mask == 1)
        ).astype('float32')

        if verbose:
            ttprint(f"Calculating zonal mean for {n_covs} layers and zonal sum for 1 mask on {year}")
        
        covs_data_idx = np.empty((n_covs, covs_data.shape[1]), dtype='float32')
        covs_data_idx[:] = covs_data[covs_idx,:]
        skb.maskDataRows(covs_data_idx, n_jobs, range(0,n_covs), mask, 0, np.nan)
        mean_data = np.empty((len(covs_idx)), dtype='float32')
        skb.nanMean(covs_data_idx, n_jobs, mean_data)
        mean_data = np.round(mean_data)

        land_data = np.empty((1, covs_data.shape[1]), dtype='float32')
        land_data[:] = mask_data[idx,:]
        skb.maskDataRows(land_data, n_jobs, [0], mask, 0, np.nan)
        sum_land = bn.nansum(land_data)
        mean_data = np.append(mean_data, sum_land)

        masked_zonal[key] = np.round(mean_data)

    zonal_df = pd.DataFrame(masked_zonal.values(), columns=covs_label)
    zonal_df['year'] = masked_zonal.keys()
    
    return zonal_df

def merge_livestock_data(polygon_samp, polygon_zonal, animals, area_col, additional_cols):
    
    polygon_zonal['polygon_idx'] = polygon_samp.index[0]
    covs_name = polygon_zonal.columns.drop([area_col,'year','polygon_idx'])
    
    polygon_zonal = polygon_zonal.set_index('polygon_idx', drop=True).merge(
        polygon_samp,
        left_index = True,
        right_index = True
    )
    
    result = []
    
    for year, rows in polygon_zonal.groupby('year'):
        
        row_cols = additional_cols + [area_col, 'year','geometry']

        for animal in animals:
            density_col = f'{animal}_density'
            rows[density_col] = rows[f'{animal}_{year}'] / rows[area_col]
            row_cols.append(density_col)

            heads_col = f'{animal}_heads'
            rows[heads_col] = rows[f'{animal}_{year}']
            row_cols.append(heads_col)

            rows.loc[np.isinf(rows[f'{animal}_density']),f'{animal}_density'] = np.nan
            rows.loc[rows[f'{animal}_density'] == 0,f'{animal}_density'] = np.nan

        rows['geometry'] = gpd.GeoSeries(rows['geometry'])
        rows['year'] = year
        result.append(rows[row_cols + list(covs_name)])

    return pd.concat(result).reset_index(drop=True)

wd = '/mnt/tupi/WRI/livestock_global_modeling/'


polygon_samples = joblib.load(f'{wd}livestock_census_raw/gpw_livestock.animals_gpw.fao.faostat.malek.2024_polygon.samples_20000101_20231231_go_epsg.4326_v1.lz4')
livestock_covs = pd.read_csv(f'{wd}/livestock_census_ard/livestock_cov.csv')
livestock_covs = livestock_covs[np.logical_or.reduce([
    livestock_covs['layername'].str.contains('bsf'),
    livestock_covs['layername'].str.contains('short.veg'),
    livestock_covs['layername'].str.contains('grassland.cropland')
])]

static_covs = [ l for l in livestock_covs[livestock_covs['type'] == 'static'].layername ]
temporal_covs = [ l for l in livestock_covs[livestock_covs['type'] == 'temporal'].layername ]

out_file = 'livestock_zonal_20250923_adhoc.pq'

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

base_path = '/mnt/tupi/WRI/livestock_global_modeling/vrt'
crs_igg = '+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

mask_ignore_val = 0
area_col = 'mask_km2'
additional_cols = ['gazID','gazName','source']
animals = ['buffalo','cattle','goat','horse','sheep']


for index, _ in polygon_samples[start_i:end_i].iterrows():

    polygon_samp = polygon_samples[polygon_samples.index == index]
    
    if processed_index is not None:
        if np.sum(processed_index == index).astype('int') > 0:
            ttprint(f"Polygon {index} already processed.")
            continue
    
    try:
        ttprint(f"Processing {polygon_samp['gazName']}")
        
        years = extract_years(polygon_samp)

        covs_info = gen_covs_info(static_covs, temporal_covs, years, base_path)
        mask_filter = covs_info['path'].str.contains('grassland.cropland')
        mask_info = covs_info[mask_filter].reset_index(drop=True)
        covs_info = covs_info[np.logical_not(mask_filter)].reset_index(drop=True)

        window = extract_window(polygon_samp, mask_info, crs_igg)
        covs_data = io.read_rasters_cpp(covs_info['path'], window=window, verbose=False, n_jobs=n_jobs)
        mask_data = io.read_rasters_cpp(mask_info['path'], window=window, verbose=False, n_jobs=n_jobs)
        polygon_mask = gen_polygon_mask(polygon_samp, mask_info, crs_igg)
        polygon_zonal = spationtemporal_masked_zonal_mean(polygon_mask, covs_data, covs_info, mask_data, mask_info, mask_ignore_val, n_jobs)

        polygon_zonal = polygon_zonal.rename(columns={'mask_sum':area_col})
        polygon_zonal[area_col] = polygon_zonal[area_col] / 100
        polygon_zonal['polygon_idx'] = polygon_samp.index[0]

        pq.write_to_dataset(
          Table.from_pandas(polygon_zonal),
          out_file,
          partition_cols=['polygon_idx'],
          compression="snappy",
          version="2.4",
        )
    except:
        print(traceback.format_exc())
        ttprint(f"Error in polygon {index}")