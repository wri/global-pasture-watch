import sys
sys.path.insert(0,'./scikit-map')
from skmap import io
from skmap.misc import find_files, ttprint,make_tempdir

import bottleneck as bn
import numexpr as ne
import numpy as np

wd = '/mnt/tupi/WRI/livestock_global_modeling/livestock_systems'
s3_path = 'gaia/gpw/livestock/'

nat_semi_grass_rule = '(natsem_grass > 0) & (natsem_grass > cultiv_grass) & (cropland <= 19) & (lst_temp_p50 > 13400) & (lst_temp_p95 < 16058)'
cultiv_grass_rule = '(cultiv_grass > 0) & (cultiv_grass >= natsem_grass) & (cropland <= 19)  & (lst_temp_p50 > 13400) & (lst_temp_p95 < 16058)'
mosaic_rule = '((cultiv_grass + natsem_grass) > 0) & (cropland > 19) & (cropland <= 37)  & (lst_temp_p50 > 13400) & (lst_temp_p95 < 16058)'
cropland_rule = '(cropland > 0) & (cropland <= 67)  & (lst_temp_p50 > 13400) & (lst_temp_p95 < 16058)'

start_year = 2000
end_year = 2022
for year in range(start_year, end_year+1):
  ttprint(f"Producing livestock land systems layers for {year}")

  raster_layers = sorted(find_files(f'{wd}/input/', f'*_{year}*.vrt')) \
    + [ f'{wd}/clm_lst_mod11a2.daytime.p50_m_1km_s0..0cm_20000101_20201231_v1.2.vrt' ] \
    + [ f'{wd}/clm_lst_mod11a2.daytime.p95_m_1km_s0..0cm_20000101_20201231_v1.2.vrt' ]
  
  raster_data = io.read_rasters_cpp(raster_layers, verbose=True, n_jobs=96)
  raster_data[np.logical_or(np.isnan(raster_data),raster_data == 255)] = 0

  total = bn.nansum(raster_data[0:3,:], axis=0)
  gt10_mask = (total > 100)

  norm_data = raster_data
  for i in range(0,3):
    norm_data[i,:][gt10_mask] = (norm_data[i,:][gt10_mask] / total[gt10_mask]) * 100

  prod_system_rules = '' \
  + f'where(({nat_semi_grass_rule}), 1,' \
  + f'where(({cultiv_grass_rule}), 2,' \
  + f'where(({mosaic_rule}), 3,' \
  + f'where(({cropland_rule}), 4,' \
  + '0 ))))'
  print(prod_system_rules)
    
  land_system_rules = 'where((prod_system == 0) | (lst_temp_p50 <= 13400) | (lst_temp_p95 >= 16058), 0, (cultiv_grass + natsem_grass + cropland))'

  local_dict={
      'cropland': norm_data[0,:],
      'cultiv_grass': norm_data[1,:],
      'natsem_grass': norm_data[2,:],
      'lst_temp_p50': norm_data[3,:],
      'lst_temp_p95': norm_data[4,:]
  }

  prod_system_data = np.empty((2, norm_data.shape[1]), dtype='float32')
  prod_system_data[0,:] = ne.evaluate(prod_system_rules, local_dict=local_dict)

  local_dict['prod_system'] = prod_system_data[0,:]
  prod_system_data[1,:] = ne.evaluate(land_system_rules, local_dict=local_dict)

  out_layers = [
    f'gpw_livestock.systems_grassland.cropland.rules_c_1km_{year}0101_{year}1231_go_epsg.4326_v1',
    f'gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{year}0101_{year}1231_go_epsg.4326_v1'
  ]
  out_idx = [0,1]
  out_dir = str(make_tempdir())
  out_s3 = [ s3_path for i in range(0,len(out_idx)) ]

  io.save_rasters_cpp(str(raster_layers[0]), prod_system_data, 
    out_layers, out_dir, out_idx, out_s3, dtype=np.uint8, nodata=0)

  ttprint(f"End")