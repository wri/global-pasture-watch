from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from eumap import parallel
from eumap.mapper import SpaceOverlay
from joblib import Parallel, delayed
from pathlib import Path
from pyarrow.dataset import dataset
from s3fs import S3FileSystem
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import random
import rasterio

def generate_year_month_list(start_date, end_date):
    result = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')

    while current_date <= datetime.strptime(end_date, '%Y-%m-%d'):
        result.append(current_date.strftime('%Y-%m'))
        current_date += relativedelta(months=1)

    return result

def worker(tile,object_path,year_month_list):
        df_list = []
    
        path = f'sampling_icesat_round{rnd}/{tile}_sampling.gpkg'

        if os.path.exists(path):
            print(f'{path} has been finished')
            return 
        for ym in year_month_list:
            subset_path = object_path + f"/tile={tile}/year={ym.split('-')[0]}/month={int(ym.split('-')[1])}"
            if '000W' in tile:
                tile_ice = tile.replace('000W','000E')
                subset_path = object_path + f"/tile={tile_ice}/year={ym.split('-')[0]}/month={int(ym.split('-')[1])}"
            elif '00S' in tile:
                tile_ice = tile.replace('00S','00N')
                subset_path = object_path + f"/tile={tile_ice}/year={ym.split('-')[0]}/month={int(ym.split('-')[1])}"
            else:
                subset_path = object_path + f"/tile={tile}/year={ym.split('-')[0]}/month={int(ym.split('-')[1])}"

            gdf_tile = gdf_tiles[gdf_tiles['TILE']==tile]
            try:
                pyarrow_dataset = dataset(source = subset_path ,format = 'parquet',filesystem=httpfs)
            # no points in that month, don't have to run the rest
            except FileNotFoundError as e:
                print(e)
                continue
            
            # filtering by med_ht and night flag
            df_default = pl.scan_pyarrow_dataset(pyarrow_dataset).filter(
                    pl.col('night_flag')==1).filter(
                pl.col('beam_strength')=='strong')

            data = df_default.collect().to_pandas()
            
            # filter by number of valid photons
            data['num_valid_photons'] = data['n_ph_20m'] + data['ph_h_canopy'].apply(lambda x: len(x.split(',')) if len(x)>0 else 0) + data['ph_h_tcanopy'].apply(lambda x: len(x.split(',')) if len(x)>0 else 0)
            data = data[data['num_valid_photons'] >=3]
            data = data[data['num_valid_photons'] <=60]   
            data['n_veg'] = data['bin0']+data['bin1']+data['bin2']+data['bin3']+data['bin4']
            data['n_signal'] = data['n_ph_20m'] - data['n_ph_te_20m'] - data['n_veg']
            data = data[data['n_signal'] <= 35]
            data = data[data['n_veg'] >= 3]
            if len(data) >= 10:
                data = data.sample(n=10,random_state=rnd)        
                data["month"] = data['start_dt'].apply(lambda x: int(x.strftime('%m')))
                data["year"] = data['start_dt'].apply(lambda x: int(x.strftime('%Y')))
            elif len(data) < 10 and len(data) > 0:
                data["month"] = data['start_dt'].apply(lambda x: int(x.strftime('%m')))
                data["year"] = data['start_dt'].apply(lambda x: int(x.strftime('%Y')))
            else:
                continue
            df_list.append(data)

            if len(df_list)>0:
                dfs = pd.concat(df_list)
                gdf = gpd.GeoDataFrame(
                    dfs, geometry=gpd.points_from_xy(dfs.lon_20m, dfs.lat_20m), crs="EPSG:4326"
                )
                gdf.to_file(f'sampling_icesat_round{rnd}/{tile}_sampling.gpkg', driver="GPKG")

httpfs = S3FileSystem(
      key='<ACCESS_KEY>',
      secret='<SECRET_KEY>',
      endpoint_url='http://192.168.49.30:8333'
   )

gdf_tiles = gpd.read_file('/mnt/fastboy/tmp/Global_tiles/ard2_final_status.gpkg')
tiles = gdf_tiles.TILE.values
object_path = 'tmp-icesat-ard/ATL08v006/atl08.v006_20181014_20230621_ga_epsg.4326_v20231130.parquet'

start_date = '2019-01-01'
end_date = '2022-12-31'
year_month_list = generate_year_month_list(start_date, end_date)

for rnd in range(10):
    rnd+=10
    os.makedirs(f'sampling_icesat_round{rnd}',exist_ok=True)

    args = [ (i,object_path,year_month_list) for i in tiles]

    for result in parallel.job(worker, args, n_jobs=-1):
        pass