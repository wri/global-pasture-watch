from skmap.mapper import LandMapper
from skmap.misc import find_files
from skmap.misc import vrt_warp, ttprint
from skmap.mapper import LandMapper
from skmap.misc import find_files
import geopandas as gpd
import numexpr as ne
import numpy as np
import traceback
from datetime import datetime
import math
import rasterio
import os

from pathlib import Path
def _model_input(tile, year, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
    prediction_layers = []
    
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
        raster_files.append(Path(l.replace('{year}', str(year))))
    
    return raster_files, dict_layers_newnames

def geo_temp(fi, day, a=37.03043, b=-15.43029):
    f =fi
    pi = math.pi 

    #math.cos((day - 18) * math.pi / 182.5 + math.pow(2, (1 - math.copysign(1, fi))) * math.pi) 
    sign = 'where(abs(fi) - fi == 0, 1, -1)'
    costeta = f"cos((day - 18) * pi / 182.5 + 2**(1 - {sign}) * pi)"

    #math.cos(fi * math.pi / 180)
    cosfi = "cos(fi * pi / 180)"
    A = cosfi

    #(1 - costeta) * abs(math.sin(fi * math.pi / 180) )
    B = f"(1 - {costeta}) * abs(sin(fi * pi / 180) )"

    x = f"(a * {A} + b * {B})"
    return ne.evaluate(x)

def inmem_calc_func(layernames, raster_data, bounds):
    import numexpr as ne
    
    lockup = dict(zip(layernames, range(0,raster_data.shape[1])))
    
    pref = 'glad.SeasConv.ard2_m_30m_s'
    suff = 'go_epsg.4326_v20230908'
    dates = ['0101_0228','0301_0430','0501_0630','0701_0831','0901_1031','1101_1231']
    bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']

    indices = {}

    for dt in dates:
        local_dict = { b: raster_data[:, :, lockup[f'{b}_{pref}_{dt}_{suff}']:lockup[f'{b}_{pref}_{dt}_{suff}']+1 ] for b in bands}
        indices[f'ndvi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) * 125 + 125', local_dict
        indices[f'ndwi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir1 * 0.004) ) / ( (nir * 0.004) + (swir1 * 0.004) ) ) * 125 + 125', local_dict
        #indices[f'savi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) )*1.5 / ( (nir * 0.004) + (red * 0.004)  + 0.5) ) * 125 + 125', local_dict
        #indices[f'msavi_{pref}_{dt}_{suff}'] = f'( (2 *  (nir * 0.004) + 1 - sqrt((2 *  (nir * 0.004) + 1)**2 - 8 * ( (nir * 0.004) - (red * 0.004) ))) / 2 ) * 125 + 125', local_dict
        #indices[f'nbr_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - ( swir2 * 0.004) ) / ( (nir * 0.004) + ( swir2 * 0.004) ) ) * 125 + 125', local_dict
        #indices[f'ndmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) -  (swir1 * 0.004)) / ( (nir * 0.004) +  (swir1 * 0.004)) ) * 125 + 125', local_dict
        #indices[f'nbr2_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - ( thermal * 0.004) ) / ( (swir1 * 0.004) + ( thermal * 0.004) ) ) * 125 + 125', local_dict
        #indices[f'rei_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - blue * 0.004)/( (nir * 0.004) + (blue * 0.004) *  (nir * 0.004)) ) * 125 + 125', local_dict
        indices[f'bsi_{pref}_{dt}_{suff}'] = f'( ( ( (swir1 * 0.004) + (red * 0.004) ) - ( (nir * 0.004) + (blue * 0.004) ) ) / ( ( (swir1 * 0.004) + (red * 0.004) ) + ( (nir * 0.004) + (blue * 0.004)) ) ) * 125 + 125', local_dict
        indices[f'ndti_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - (swir2 * 0.004) )  / ( (swir1 * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
        #indices[f'ndsi_{pref}_{dt}_{suff}'] = f'( ( (green * 0.004) -  (swir1 * 0.004) ) / ( (green * 0.004) +  (swir1 * 0.004) ) ) * 125 + 125', local_dict
        #indices[f'ndsmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir2 * 0.004) )  / ( (nir * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
        indices[f'nirv_{pref}_{dt}_{suff}'] = f'( ( ( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) - 0.08) *  (nir * 0.004) ) * 125 + 125', local_dict
        indices[f'evi_{pref}_{dt}_{suff}'] = f'( 2.5 * ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + 6 * (red * 0.004) - 7.5 * (blue * 0.004) + 1) ) * 125 + 125', local_dict
        indices[f'fapar_{pref}_{dt}_{suff}'] = f'( ((( (( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) )) - 0.03) * (0.95 - 0.001)) / (0.96 - 0.03)) + 0.001 ) * 125 + 125', local_dict
    
    result = [ raster_data ]
    bcf_local_dict = {}
    min_val, max_val = 0, 250
    for index, (expr, local_dict) in indices.items():
        print(f'Calculating {index}')
        newdata = ne.evaluate(expr, local_dict=local_dict).round()
        newdata[np.logical_or(newdata < min_val, newdata == -np.inf)] = 0
        newdata[np.logical_or(newdata > max_val, newdata == np.inf)] = 250
        result.append(newdata)
        layernames.append(index)
        if 'ndvi_' in index:
            _index = index.replace(f'_{pref}','').replace(f'_{suff}','')
            bcf_local_dict[_index] = newdata
            print(index, newdata.shape)
            
    expr = f'( where( ndvi_0101_0228 <= 169, 100, 0) + where( ndvi_0301_0430 <= 169, 100, 0) + ' + \
           f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) + ' + \
           f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) ) / 6'                
    index = f'bsf_{pref}_{suff}'
    print(f'Calculating {index}')
    newdata = ne.evaluate(expr, local_dict=bcf_local_dict).round()
    result.append(newdata)
    layernames.append(index)
        
    with rasterio.open(vrt_files[0]) as ds:
        pixel_size = ds.transform[0]
        lon = np.arange(0, 4000)
        lat = np.arange(0, 4000)

        lon_grid, lat_grid = ds.transform * np.meshgrid(lon, lat)

        elev_corr = 0.006 * raster_data[:, :, lockup['dtm.bareearth_ensemble_p10_30m_s_2018_go_epsg4326_v20230210'] ] * 0.1

        for m in range(1,13):
            doy = (datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday)
            max_temp_name = f'clm_lst_max.geom.temp_m_30m_s_m{m}' 
            min_temp_name = f'clm_lst_min.geom.temp_m_30m_s_m{m}'
            print(f"Adding {max_temp_name} & {min_temp_name}")
            print(lat_grid.shape, elev_corr.shape)
            layernames.append(max_temp_name)
            result.append(np.stack( [ ((geo_temp(lat_grid, day=doy, a=37.03043, b=-15.43029) - elev_corr) * 100).round() ], axis=2))
            layernames.append(min_temp_name)
            result.append(np.stack( [ ((geo_temp(lat_grid, day=doy, a=24.16453, b=-15.71751) - elev_corr) * 100).round() ], axis=2))

    print(raster_data.shape)
    raster_data = np.concatenate(result, axis=-1)
    print(raster_data.shape)
    
    return layernames, raster_data

fn_landmapper =  './model_v20240210/landmapper_100.lz4' #'global_model_v1.lz4'
m = LandMapper.load_instance(fn_landmapper)
static_raster = find_files('./static', '*.vrt')
tiles = gpd.read_file('ard2_final_status.gpkg')

#_tiles = ['003W_57N','006E_45N','016E_12S','029E_51N','047W_11S','055W_17S','055W_28S','056W_10S','061W_28S','075E_25N','081E_60N','081E_60N','091W_17N','101E_28N','102W_43N','103E_46N','115E_28S','121E_04S','145E_18S','146W_62N','147E_25S']
#_years = [2020, 2000, 2005, 2010, 2015, 2016, 2017, 2018, 2019, 2022]

_tiles = ['055W_17S']
#_tiles = ['055W_17S', '091W_17N', '003W_57N','006E_45N','016E_12S','029E_51N','047W_11S','055W_28S','056W_10S','061W_28S','075E_25N','081E_60N','081E_60N','101E_28N','102W_43N','103E_46N','115E_28S','121E_04S','145E_18S','146W_62N','147E_25S'] #_tiles = ['003W_57N','006E_45N','016E_12S','029E_51N','047W_11S','055W_28S','056W_10S','061W_28S','075E_25N','081E_60N','081E_60N','101E_28N','102W_43N','103E_46N','115E_28S','121E_04S','145E_18S','146W_62N','147E_25S']
_years = [2020] #[2001,2002,2003,2004,2006,2007,2008,2009,2011,2012,2013,2014,2001]

for year in _years: 
  for tile in _tiles:
    try:
      ttprint(f"Predicting {tile} {year}")
      bounds = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds

      raster_files, dict_layers_newnames = _model_input(tile, year)
      raster_files += static_raster

      vrt_files = vrt_warp(raster_files, dst_crs='EPSG:4326', tr=0.00025, te = bounds)
      vrt_files = [ Path(f) for f in vrt_files ]

      m.predict(fn_layers=vrt_files, fn_output=f'prediction_v2024_gpw_ultimate/{tile}/pasture.class_{year}_{tile}.tif', 
                allow_additional_layers=True, 
                dict_layers_newnames=dict_layers_newnames, 
                inmem_calc_func=inmem_calc_func, 
                n_jobs_io=96
      )
      
      for vrt_file in vrt_files:
        os.unlink(vrt_file)

    except:
      traceback.print_exc()
      print(f"Error {tile} {year}")
      #break