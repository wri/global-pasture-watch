from eumap.misc import find_files, GoogleSheet, ttprint
from pathlib import Path
import geopandas as gpd 
import pandas as pd

input_dir = 'GPW_review_Noisy_Areas_points'

samples = []
files = find_files(input_dir, '*.csv.gz')
ttprint(f"Reading {len(files)} *.csv.gz files")
for file in files:
  data = pd.read_csv(file)
  tile_id = str(Path(file).name).split('_')[2][5:]
  data['tile_id'] = tile_id
  samples.append(data)

samples = pd.concat(samples)

points = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['x'], samples['y']))
points = points.set_crs('EPSG:4326')

tiles = gpd.read_file('ard2_final_status.gpkg')
points_tiles = points.sjoin(tiles[['TILE','geometry']], how="left")

#points_tiles.to_file('global_samples.gpkg')

mask = points_tiles['date'].str[:4].str.contains('-')
points_tiles.loc[mask,'date'] = points_tiles[mask]['date'].str[6:10] + '-' + points_tiles[mask]['date'].str[3:5] + '-' + points_tiles[mask]['date'].str[0:2] 

points_tiles['ref_date'] = pd.to_datetime(points_tiles['date'])
points_tiles = points_tiles.rename(columns={'tile_id':'vi_tile_id','TILE':'glad_tile_id'})
points_tiles = points_tiles[(points_tiles['ref_date'] > '2000-01-01') & (points_tiles['ref_date'] <= '2022-12-31')]
points_tiles = points_tiles.drop(columns=['date'])

points_tiles.drop(columns=['geometry']).to_parquet('gpw_samples_reviewed_v20240415.pq')