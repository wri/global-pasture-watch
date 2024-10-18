from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
import numpy as np
import os
from pyproj import Proj, transform
from eumap.misc import find_files, ttprint
from eumap import parallel
import traceback

def _tile_id(samp_path):
	return str(Path(samp_path).name).split('_')[0]

def add_date(args_live, in_dir_past, col_dt):
	for a in args_live:
		live_path = a[0]
		tile_id = _tile_id(live_path)
		files = find_files(in_dir_past, f'{tile_id}_*.gpkg')
		if len(files) > 0:
			live_ds = gpd.read_file(live_path)

			if col_dt not in live_ds:
				past_path = files[0]

				samp_ds = gpd.read_file(past_path)
				date = samp_ds[col_dt][~samp_ds[col_dt].isnull()].iloc[0]

				ttprint(f"Adding date {date} to {live_path}")
				live_ds[col_dt] = date
				live_ds.to_file(live_path, drive="GPKG")

def convert_to_samples(samp_path, col_dt_suff, col_cls_suff, col_mvp_name, class_values, out_dir, tr_m, tr_d, samp_type, scale, 
	col_agg_name, tiles_path = None):
	
	samp_ds = gpd.read_file(samp_path)
	tile_id = _tile_id(samp_path)
	prefix_list = ['google', 'bing']

	dt_cls_array = [ 	( f'{prefix}_{col_dt_suff}', f'{prefix}_{col_cls_suff}', prefix) 
										for prefix in prefix_list ]
	
	csv_files = []

	for (col_dt, col_cls, prefix) in dt_cls_array:

		try:
			_samp_ds = samp_ds[~pd.isnull(samp_ds[col_cls])]
			print(_samp_ds.shape)
			row = _samp_ds.iloc[0]
			

			if row[col_dt] is None:
				ttprint(f"Skipping {prefix} (empty date) for {samp_path}.")
				continue

			if row[col_dt] == '2000-01-01':
				ttprint(f"Skipping date=2000-01-01 for {samp_path}.")
				continue				 

			if row[f'{prefix_list[0]}_{col_dt_suff}'] == row[f'{prefix_list[1]}_{col_dt_suff}']:
				ttprint(f"Skipping Bing (same image) for {samp_path}.")
				continue

			if tiles_path is not None:
				tiles_ds = gpd.read_file(tiles_path)
				bounds = tiles_ds[tiles_ds['tile_id'] == int(tile_id)].iloc[0]['geometry'].bounds
			else:
				bounds = _samp_ds.total_bounds
			
			if col_cls is not None:
				_samp_ds['value'] = _samp_ds[col_cls].map(class_values)
			else:
				_samp_ds['value'] = 1
			
			_samp_ds['geometry'] = _samp_ds['geometry'].centroid
			date = _samp_ds[col_dt][~_samp_ds[col_dt].isnull()].iloc[0]
			daten = date.replace('-','.')
			year = date.split('-')[0]

			_out_dir = Path(out_dir).joinpath(f'{year}')
			_out_dir.mkdir(exist_ok=True)
			
			points_vec = _out_dir.joinpath(f'lcv_{samp_type}_tile.{tile_id}_{daten}.gpkg')
			samples_csv = _out_dir.joinpath(f'lcv_{samp_type}_tile.{tile_id}_{daten}.csv.gz')

			if samples_csv.exists():
				return samples_csv

			if not points_vec.exists():
				_samp_ds[['value', 'geometry']].to_file(points_vec, driver='GPKG')
			
			weigth_arr, lon_arr, lat_arr, val_arr, dt_arr = [], [], [], [], []
			
			for value in list(_samp_ds['value'].unique()):
				if not np.isnan(value):
					value = int(value)
					
					points_ras1 = _out_dir.joinpath(f'lcv_{samp_type}_tile.{tile_id}.{value}_{daten}_1.tif')
					points_ras2 = _out_dir.joinpath(f'lcv_{samp_type}_tile.{tile_id}.{value}_{daten}_2.tif')
					points_ras3 = _out_dir.joinpath(f'lcv_{samp_type}_tile.{tile_id}.{value}_{daten}.tif')

					if not points_ras3.exists():
						rast_cmd1 = "gdal_rasterize -a_nodata 0 -add -burn 1 " \
								+ f' -where "value = {value}" ' \
								+ f' -te {" ".join([str(b) for b in bounds])}' \
								+ f' -co COMPRESS=deflate -ot Byte -tr {tr_m} {tr_m}' \
								+ f' {points_vec} {points_ras1}'

						pct_scale = 1 # Pixel counting
						if scale > 1: 
							pct_scale = 100 # Pixel percentage

						rast_cmd2 = 'gdal_calc.py --NoDataValue 0 --co="COMPRESS=deflate"' \
						+ f' --calc="(A/{scale})*{pct_scale}" -A {points_ras1} --outfile={points_ras2}'

						crs = 'EPSG:4326'
						rast_cmd3 = f"gdalwarp -overwrite -t_srs '{crs}' -tr {tr_d} {tr_d}" \
						f' {points_ras2} {points_ras3}'

						os.system(rast_cmd1)
						os.system(rast_cmd2)
						os.system(rast_cmd3)

						points_ras1.unlink()
						points_ras2.unlink()
					
					ds = rasterio.open(points_ras3)
					samp_data = ds.read(1)
					
					lat = np.arange(0.5, samp_data.shape[0] + 0.5, dtype='float32')
					lon = np.arange(0.5, samp_data.shape[1] + 0.5, dtype='float32')
					lon_grid, lat_grid = ds.transform * np.meshgrid(lon, lat)
					
					samp_mask = (samp_data > 0)
					samp_n = samp_data[samp_mask].shape[0]
					
					weigth_arr += list(samp_data[samp_mask])
					#print(samp_mask.shape)
					#print(lon_grid.shape)
					#print(samp_data.shape)
					lon_arr += list(lon_grid[samp_mask])
					lat_arr += list(lat_grid[samp_mask])
					
					for i in range(0,samp_n):
						val_arr.append(value)
						dt_arr.append(date)
			
			result = pd.DataFrame(
				np.column_stack([val_arr, lon_arr, lat_arr, weigth_arr, dt_arr]), 
				columns=['class', 'x', 'y', col_agg_name, 'date']
			)

			#result['lat_epsg4326'], result['lon_epsg4326'] = result['x'], result['y']
			result['tile_id'] = tile_id
			#result['crs'] = crs
			result['imagery'] = prefix.title()
			
			result.to_csv(samples_csv, compression='gzip', index=False)
			points_vec.unlink()
			csv_files.append(samples_csv)

		except:
			traceback.print_exc()
			print(f"ERROR in {samp_path} ({prefix})")

	return csv_files

#convert_to_samples(samp_path, mvp_path, col_dt, col_cls, col_mvp_name, class_values, out_dir, tr)

col_dt = 'image_start_date'
col_cls = 'class'

in_dir_past = './GPW_review_Noisy_Areas/'
out_dir_past = 'GPW_review_Noisy_Areas_points'

past_prefix = 'pasture.samples'
past_col_agg  = 'class_pct'
past_scale = 36

tiles_path = './lcv_pastures_fscs.tiles_epsg.3857.gpkg'
col_mvp_name = 'name'
class_values = {
	'SEEDED GRASS': 1,
	'NATURAL OR SEMI-NATURAL GRASS': 2,
	'OTHERS': 3
}

tr_m, tr_d = 60, 0.000500000000000

# TODO: Threat invalid classes
invalid_classes = {
 'NATURAL OR SEMI-NATURAL GRASSNATURAL OR SEMI-NATURAL GRASS' : 'NATURAL OR SEMI-NATURAL GRASS',
 'NATURAL OR SEMI-NATURAL GRASS ' : 'NATURAL OR SEMI-NATURAL GRASS',
 'NATURAL OR SEMI-NATURAL GRASSNATURAL OR SEMI-NATURAL GRASSNATURAL OR SEMI-NATURAL GRASSNATURAL OR SEMI-NATURAL GRASS' : 'NATURAL OR SEMI-NATURAL GRASS',
 'NATURAL OR SEMI-NATURAL GRASSV'  : 'NATURAL OR SEMI-NATURAL GRASS',
 'OTHER' : 'OTHERS',
 '' : 'NOT DEFINED',
 'NULL': 'NOT DEFINED'
}

args_past = [
	(f, col_dt, col_cls, col_mvp_name, class_values, out_dir_past, tr_m, tr_d, past_prefix, past_scale, past_col_agg)
	for f in find_files(in_dir_past, '*.gpkg')
]

for result in parallel.job(convert_to_samples, args_past, n_jobs=40):
	print(result)