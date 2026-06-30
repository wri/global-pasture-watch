from eumap.raster import read_rasters, save_rasters
from eumap.misc import find_files, ttprint
from eumap.parallel import TilingProcessing, job
import numpy as np
from pathlib import Path
import bottleneck as bn
import traceback

tiling_system_fn = '/mnt/tupi/hilda_plus/tiles.gpkg'
base_raster_fn = 'http://192.168.1.57:9000/global/lcv/lcv_land.cover_esacci.lc.l4_c_250m_s0..0cm_2020_v1.0.tif'
tile_processing = TilingProcessing(tiling_system_fn = tiling_system_fn, base_raster_fn=base_raster_fn, verbose=False)
raster_files = [ f'http://192.168.1.57:9000/global/lcv/lcv_land.cover_esacci.lc.l4_c_250m_s0..0cm_{year}_v1.0.tif' for year in range(1993,2021) ] 

short_veg_classes = [30,40,100,110,120,121,122,130,150,151,152,153,180]
out_dir = Path('/mnt/tupi/cci_short_veg/')

def run(idx, tile, window, raster_files, short_veg_classes, out_dir):
    try:
        short_veg_classes = np.array(short_veg_classes).astype('float32')
        data, _ = read_rasters(raster_files=raster_files[0:1], spatial_win=window, n_jobs=10)
        if(bn.nanmin(data) ==  210):
            ttprint(f'Skipping {idx}')
            return False
        else:
            data, _ = read_rasters(raster_files=raster_files, spatial_win=window, n_jobs=1, dtype='float32')
            result = bn.nansum((np.isin(data, short_veg_classes)).astype('int8'), axis = -1)
            #print(np.unique(data), np.unique(result))
            result = np.stack([(result /  data.shape[2]) * 100], axis=-1)
            #result = np.stack([result], axis=-1)
            
            out_files = out_dir.joinpath(f'lcv_land.cover_esacci.lc.short.veg_c_250m_s0..0cm_2020_v1.0').joinpath(f'{idx}.tif')
            ttprint(f'Saving {out_files}')
            save_rasters(raster_files[0], [ out_files ], result, spatial_win=window, n_jobs=1)
            return True
    except:
        traceback.format_exc()
        print(f'ERROR {idx}')

missing_tiles = [6,372,705,853,1010,1155,1232,1343,1716,1761,1895,2040,2160,2444,2609,2824,2947,3086,3254]
    
x = tile_processing.process_multiple(missing_tiles, run, raster_files, short_veg_classes, out_dir, use_threads=False, max_workers=1)
#x = tile_processing.process_all(run, raster_files, short_veg_classes, out_dir, use_threads=False, max_workers=96)