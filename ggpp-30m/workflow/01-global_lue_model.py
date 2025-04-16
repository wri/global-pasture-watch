from osgeo import gdal
from skmap.misc import make_tempdir, ttprint
from skmap.misc import ttprint
from skmap.misc import ttprint
import bottleneck as bn
import bottleneck as bn
import geopandas as gpd
import numexpr as ne
import numpy as np
import numpy as np
import os
import os
import pandas as pd
import requests
import skmap_bindings
import skmap_bindings
import time
import traceback

os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
n_threads = 96
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}' # export VECLIB_MAXIMUM_THREADS=4

import geopandas as gpd
import numpy as np
import numpy as np
import pandas as pd
import random
import requests
import skmap_bindings as sb
import sys
import time
import traceback

GAIA_MAX_IP = 46

def _landsat_files(tile_id, bands, y0, y1):
    month_start = ['0101','0301','0501','0701','0901','1101']
    month_end = ['0228' ,'0430' ,'0630' ,'0831' ,'1031' ,'1231']
    raster_files = []
    for band in bands:
        for year in range(y0,y1+1):
            for bm in range(len(month_start)):
                raster_files += [f"http://192.168.49.{random.randint(30,GAIA_MAX_IP)}:8333/prod-landsat-ard2/{tile_id}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_{year}{month_start[bm]}_{year}{month_end[bm]}_go_epsg.4326_v20230908.tif"]
    return raster_files


def _modis_files(y0, y1):
    mod11a2p50_files = []
    PAR_toa_files = []
    month_start = ['01.01','02.01','03.01','04.01','05.01','06.01','07.01','08.01','09.01','10.01','11.01','12.01']
    month_end = ['01.31','02.28','03.31','04.30','05.31','06.30','07.31','08.31','09.30','10.31','11.30','12.31']
    month_id = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in range(y0,y1+1):
        for bm in range(len(month_start)):
            mod11a2p50_files += [f"http://192.168.49.{random.randint(30,GAIA_MAX_IP)}:8333/gpw/vrt/gpp/clm_lst_mod11a2.daytime_p50_30m_s0..0cm_{min(year,2021)}.{month_start[bm]}..{min(year,2021)}.{month_end[bm]}_v1.2.vrt"]
            PAR_toa_files += [f"http://192.168.49.{random.randint(30,GAIA_MAX_IP)}:8333/gpw/vrt/gpp/par_toa_all.sky.adj_mean_syn1deg_1deg_{year}{month_id[bm]}_global_epsg4326_4.1.vrt"]
    return mod11a2p50_files + PAR_toa_files


def _processed(tile_id, out_prefix, out_files):
    for out_file in out_files:
        url = f'http://192.168.49.{random.randint(30, GAIA_MAX_IP)}:8333/{out_prefix}/{tile_id}/{out_file}.tif'
        r = requests.head(url)
        if r.status_code != 200:
            return False
    return True

grassland_tiles = pd.read_csv('/mnt/slurm/jobs/wri_gpp/grassland_tiles.csv')
def _grassland_files(tile_id, y0, y1):
    path = grassland_tiles.loc[grassland_tiles['tile'] == tile_id,'path'].iloc[0]
    return [f"http://192.168.49.{random.randint(30, GAIA_MAX_IP)}:8333/tmp-gpw/{path}/{tile_id}/gpw_grassland_rf.savgol.bthr_c_30m_{year}0101_{year}1231_go_epsg.4326_v1.tif"
        for year in range(y0,y1 + 1)]

def run_gpp_tile(tile_id, min_x, max_y):
    print(f"Starting tile {tile_id}")

    gdal_opts = {
        'GDAL_HTTP_VERSION': '1.0',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif'
    }
    bands_list = [1,]
    bands = ['red', 'nir', 'swir1']
    y0, y1 = 2000, 2022
    n_years = y1 - y0 + 1
    n_img_per_year = 6
    n_s = n_years * n_img_per_year
    x_off, y_off = 2, 2
    x_size, y_size = 4000, 4000
    n_pix = x_size * y_size

    landsat_files = _landsat_files(tile_id, bands, y0, y1)
    modis_files = _modis_files(y0, y1)

    grass_data = np.empty((n_years, n_pix), dtype=np.float32)
    filtered_grass_data = np.empty((n_years, n_pix), dtype=np.float32)
    skip_grass = False
    try:
        grass_files = _grassland_files(tile_id, y0=y0, y1=y1)
        sb.readData(grass_data, n_threads, grass_files, range(len(grass_files)), 0, 0, x_size, y_size, [1,], gdal_opts, 255., 0.)
    except:
        ttprint(f"Tile {tile_id} - Skipping grassland for this tile.")
        skip_grass = True
    if not(skip_grass):
        sb.slidingWindowClassMode(grass_data, n_threads, filtered_grass_data, 5)
    s3_prefix = 'tmp-gpw/gpp_v20241126'
    bimonth_start = ['0101', '0301', '0501', '0701', '0901', '1101']
    bimonth_end = ['0228', '0430', '0630', '0831', '1031', '1231']
    out_files_land = []
    out_files_grass = []
    out_files_land_year = []
    out_files_grass_year = []
    out_files_grass_filt_maks = []

    for year in range(y0, y1 + 1):
        out_files_grass_filt_maks.append(f'gpw_grassland_rf.savgol.bthr.mode5_c_30m_{year}0101_{year}1231_go_epsg.4326_v1')
        out_files_land_year.append(f'gpw_ugpp_lue.model_m_30m_s_{year}0101_{year}1231_go_epsg.4326_v1.2')
        out_files_grass_year.append(f'gpw_gpp.grass_lue.model_m_30m_s_{year}0101_{year}1231_go_epsg.4326_v1.2')  
        for bm in range(n_img_per_year):
            out_files_land.append(f'gpw_ugpp.daily_lue.model_m_30m_s_{year}{bimonth_start[bm]}_{year}{bimonth_end[bm]}_go_epsg.4326_v1.2')
            out_files_grass.append(f'gpw_gpp.daily.grass_lue.model_m_30m_s_{year}{bimonth_start[bm]}_{year}{bimonth_end[bm]}_go_epsg.4326_v1.2')
    if skip_grass:
        out_files_year = out_files_land_year
        out_files_bimonth = out_files_land
        out_files = out_files_year + out_files_bimonth
    else:
        out_files_year = out_files_land_year + out_files_grass_year
        out_files_bimonth = out_files_land + out_files_grass
        out_files = out_files_year + out_files_bimonth + out_files_grass_filt_maks
        
    if not _processed(tile_id, s3_prefix, out_files):
                
        start = time.time()
        x_off, y_off = 2, 2
        x_size, y_size = 4000, 4000
        n_pix = x_size * y_size
        array_landsat = np.zeros((n_s*len(bands), n_pix), dtype=np.float32)
        start = time.time()
        skmap_bindings.readData(array_landsat, n_threads, landsat_files, range(len(landsat_files)), x_off, y_off, x_size, y_size, bands_list, gdal_opts)        
        ttprint(f"Tile {tile_id} - Reading landsat data: {(time.time() - start):.2f} s")

        start = time.time()
        array_mod1 = np.zeros((n_s, n_pix), dtype=np.float32)
        array_mod2 = np.zeros((n_s, n_pix), dtype=np.float32)
        array_par1 = np.zeros((n_s, n_pix), dtype=np.float32)
        array_par2 = np.zeros((n_s, n_pix), dtype=np.float32)
        gidal_ds = gdal.Open(modis_files[0])
        gt = gidal_ds.GetGeoTransform()
        gti = gdal.InvGeoTransform(gt)
        x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
        x_off_s, y_off_s = int(x_off_s), int(y_off_s)
        skmap_bindings.readData(array_mod1, n_threads, [modis_files[i] for i in range(0,276,2)], range(n_s), x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        skmap_bindings.readData(array_mod2, n_threads, [modis_files[i] for i in range(1,276,2)], range(n_s), x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        skmap_bindings.readData(array_par1, n_threads, [modis_files[i] for i in range(276,552,2)], range(n_s), x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        skmap_bindings.readData(array_par2, n_threads, [modis_files[i] for i in range(277,552,2)], range(n_s), x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        ttprint(f"Tile {tile_id} - Reading MODIS data: {(time.time() - start):.2f} s")

        start = time.time()
        local_dict = {
            'mod1': array_mod1,
            'mod2': array_mod2,
            'par1': array_par1,
            'par2': array_par2
        }
        modis_bimonthly = ne.evaluate("(mod1 + mod2) / 2", local_dict = local_dict)
        par_bimonthly = ne.evaluate("(par1 + par2) / 2", local_dict = local_dict)
        ttprint(f"Tile {tile_id} - MODIS bimonthly aggregate: {(time.time() - start):.2f} s")

        local_dict = {
            'red': array_landsat[0:n_s,:], 
            'nir': array_landsat[n_s:2*n_s,:],
            'swir': array_landsat[2*n_s:3*n_s,:],
            'lst': modis_bimonthly,
            'par': par_bimonthly,
            'fpar_min': 0.001,
            'fpar_max': 0.95,
            'ndvi_min': 0.03,
            'ndvi_max': 0.96,
            't_min': 0,
            't_max': 48,
            't_opt': 20.33
        }

        lst_form = f'(lst*0.02)-273.15'
        par_form = f'0.0864*par'
        ndvi_form = f'((nir/250.0 - red/250.0) / (nir/250.0 + red/250.0))'
        fpar_form = f'(((({ndvi_form} - ndvi_min)*(fpar_max - fpar_min))/(ndvi_max - ndvi_min)) + fpar_min)'
        lswi_form = f'((nir/250.0 - swir/250.0) / (nir/250.0 + swir/250.0))'
        t_scalar_form = f'((({lst_form} - t_max) * ({lst_form} - t_min)) / (({lst_form} - t_max) * ({lst_form} - t_min) - ({lst_form} - t_opt)**2))'

        start = time.time()
        lswi = ne.evaluate(lswi_form, local_dict = local_dict)
        lswi_max_stack = np.empty(lswi.shape, dtype='float32')
        for pos in range (0,n_s,n_img_per_year):
            lswi_max = bn.nanmax(lswi[pos:(pos+n_img_per_year),:],axis=0)
            lswi_max = np.repeat(lswi_max.reshape(1,-1),n_img_per_year,axis=0)                    
            lswi_max_stack[pos:(pos+n_img_per_year),:] = lswi_max             
            lswi_max = None

        local_dict['lswi_max_stack'] = lswi_max_stack
        ttprint(f"Tile {tile_id} - Landsat LSWI: {(time.time() - start):.2f} s")

        w_scalar_form = f'(1+{lswi_form})/(1+lswi_max_stack)'
        lue_factor = 1
        gpp_form = f'(({par_form} * {fpar_form} * {w_scalar_form} * {t_scalar_form})*{lue_factor}*10)'

        start = time.time()
        gpp = ne.evaluate(gpp_form, local_dict = local_dict)
        fpar = ne.evaluate(fpar_form, local_dict = local_dict)
        gpp[fpar<0]=0
        gpp[gpp<0]=0
        ttprint(f"Tile {tile_id} - GPP calculation: {(time.time() - start):.2f} s")
        
        start = time.time()
        gpp_landmasked = gpp.astype(np.float32)
        gpp_grassmasked = np.empty((n_s, x_size * y_size), dtype=np.float32)
        land_mask = np.empty((1, x_size * y_size), dtype=np.float32)
        sb.extractArrayRows(gpp_landmasked, n_threads, gpp_grassmasked, range(n_s)) # Copy the data
        land_files = [f'http://192.168.49.30:8333/gpw/landmask/{tile_id}.tif']
        sb.readData(land_mask, n_threads, land_files, range(len(land_files)), x_off, y_off, x_size, y_size, [1,], gdal_opts, 255., 0.)
        sb.maskDataRows(gpp_landmasked, n_threads, range(gpp_landmasked.shape[0]), land_mask, 0., np.nan)
        if not(skip_grass):
            for year in range(n_years):
                year_grass_mask = np.empty((1, x_size * y_size), dtype=np.float32)
                sb.extractArrayRows(filtered_grass_data, n_threads, year_grass_mask, [year])
                sb.maskDataRows(gpp_grassmasked, n_threads, range(n_img_per_year*year, n_img_per_year*(year+1)), year_grass_mask, 0., np.nan)
        print(f"Tile {tile_id} - Masking GPP: {(time.time() - start):.2f} s")

        start = time.time()
        gpp_landmasked_t = np.empty((x_size * y_size, n_s), dtype=np.float32)
        gpp_grassmasked_t = np.empty((x_size * y_size, n_s), dtype=np.float32)
        gpp_landmasked_yearly = np.empty((n_years, x_size * y_size), dtype=np.float32)
        gpp_grassmasked_yearly = np.empty((n_years, x_size * y_size), dtype=np.float32)
        gpp_landmasked_yearly_t = np.empty((x_size * y_size, n_years), dtype=np.float32)
        gpp_grassmasked_yearly_t = np.empty((x_size * y_size, n_years), dtype=np.float32)
        sb.transposeArray(gpp_landmasked, n_threads, gpp_landmasked_t)
        sb.transposeArray(gpp_grassmasked, n_threads, gpp_grassmasked_t)
        sb.averageAggregate(gpp_landmasked_t, n_threads, gpp_landmasked_yearly_t, n_img_per_year)
        sb.averageAggregate(gpp_grassmasked_t, n_threads, gpp_grassmasked_yearly_t, n_img_per_year)
        sb.transposeArray(gpp_landmasked_yearly_t, n_threads, gpp_landmasked_yearly)
        sb.transposeArray(gpp_grassmasked_yearly_t, n_threads, gpp_grassmasked_yearly)
        sb.offsetAndScale(gpp_landmasked_yearly, n_threads, 0., 36.5)
        sb.offsetAndScale(gpp_grassmasked_yearly, n_threads, 0., 36.5)
        print(f"Tile {tile_id} - GPP annual comulative aggregation for land and grassland: {(time.time() - start):.2f} s")
        
        start = time.time()
        tmp_dir = str(make_tempdir(tile_id))
        no_data_uint8 = 255
        no_data_uint16 = int(65000)
        compression_command_uint8 = f"gdal_translate -a_nodata {no_data_uint8} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
        compression_command_uint16 = f"gdal_translate -a_nodata {no_data_uint16} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
        sb.maskNan(gpp_landmasked_yearly, n_threads, range(gpp_landmasked_yearly.shape[0]), no_data_uint16)
        sb.maskNan(gpp_grassmasked_yearly, n_threads, range(gpp_grassmasked_yearly.shape[0]), no_data_uint16)
        sb.maskNan(gpp_landmasked, n_threads, range(gpp_landmasked.shape[0]), no_data_uint8)
        sb.maskNan(gpp_grassmasked, n_threads, range(gpp_grassmasked.shape[0]), no_data_uint8)
        gpp_yearly = np.empty((n_years*2, x_size * y_size), dtype=np.float32)
        sb.expandArrayRows(gpp_landmasked_yearly, n_threads, gpp_yearly, range(0, n_years))
        sb.expandArrayRows(gpp_grassmasked_yearly, n_threads, gpp_yearly, range(n_years, n_years*2))
        gpp_bimonth = np.empty((n_s*2, x_size * y_size), dtype=np.float32)
        sb.expandArrayRows(gpp_landmasked, n_threads, gpp_bimonth, range(0, n_s))
        sb.expandArrayRows(gpp_grassmasked, n_threads, gpp_bimonth, range(n_s, n_s*2))

        base_files = landsat_files + landsat_files
        out_s3 = [ f"g{random.randint(1,GAIA_MAX_IP-30)}/{s3_prefix}/{tile_id}" for o in out_files_bimonth ]
        sb.writeUInt16Data(gpp_yearly, n_threads, gdal_opts, base_files[0:len(out_files_year)], tmp_dir, out_files_year, range(len(out_files_year)),
                    x_off, y_off, x_size, y_size, no_data_uint16, compression_command_uint16, out_s3)
        sb.writeByteData(gpp_bimonth, n_threads, gdal_opts, base_files[0:len(out_files_bimonth)], tmp_dir, out_files_bimonth, range(len(out_files_bimonth)),
                    x_off, y_off, x_size, y_size, no_data_uint8, compression_command_uint8, out_s3)
        if not(skip_grass):
            sb.writeByteData(filtered_grass_data, n_threads, gdal_opts, base_files[0:len(out_files_grass_filt_maks)], tmp_dir, out_files_grass_filt_maks, range(len(out_files_grass_filt_maks)),
                        x_off, y_off, x_size, y_size, no_data_uint8, compression_command_uint8, out_s3)
        print(f"Tile {tile_id} - Saving data: {(time.time() - start):.2f} s")

        ttprint(f"Tile {tile_id} - Done. ")
        print(f"Check mc ls {out_s3[0]}/{out_files_year[0]}.tif")
    else:
        ttprint(f"Tile {tile_id} - Already exists. Skipping. ")   

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

with open('/mnt/slurm/jobs/wri_gpp/ard2_all_ids.csv', "r") as file:
    tiles_ids = [line.strip() for line in file]
tiles_ids = tiles_ids[start_tile:end_tile]
tiles_fn = '/mnt/bender/wri_gpp_prod/ard2_final_status.gpkg'
tiles = gpd.read_file(tiles_fn)

for tile_id in tiles_ids:
    try:
        min_x, _, _, max_y = tiles[tiles['TILE'] == tile_id].iloc[0].geometry.bounds
        run_gpp_tile(tile_id, min_x, max_y)
    except:
        tb = traceback.format_exc()
        ttprint(f"Tile {tile_id} - Error")
        ttprint(tb)
        continue
        
