from numba import njit
from pathlib import Path
from pyarrow import Table
from rasterio.windows import Window
from skmap.io import read_rasters_cpp
from skmap.misc import ttprint

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

@njit(parallel = True, fastmath = True)
def unique_counts_bitwise(arr):
  
  packed = (arr[0].astype(np.int32) << 16) | (arr[1].astype(np.int32) << 8) | arr[2].astype(np.int32)
  sorted_packed = np.sort(packed)

  n = len(sorted_packed)
  uniques = [sorted_packed[0]]
  counts = [1]
  for i in range(1, n):
    if sorted_packed[i] == sorted_packed[i - 1]:
      counts[-1] += 1
    else:
      uniques.append(sorted_packed[i])
      counts.append(1)

  packed_arr = np.array(uniques)
  return packed_arr >> 16 & 0xFF, packed_arr >> 8 & 0xFF, packed_arr & 0xFF, np.array(counts)



raster_files = [
    'gpw_short.veg.height_egbt.trend.mk_m_30m_s_20000101_20221231_go_esri.54008_v20240809.vrt',
    'land.cover_ifad.mcdt_c_30m_s_20200101_20201231_go_esri.54008_v1.1.vrt',
    'continent.mask_c_30m_s_20200101_20201231_go_esri.54008_v20260629.vrt',
]

gdal_opts = {
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_USE_HEAD': 'NO'
}

print(sys.argv)

arg_fn = sys.argv[1] #  '/mnt/atlas/WRI/args.in'
i0 = int(sys.argv[2])
i1 = int(sys.argv[3])
out_file = sys.argv[4] # '/mnt/atlas/WRI/gpw_sampling.design.pq'

args = np.loadtxt(arg_fn, dtype=str)

ttprint("Start")

for arg in args[i0:i1]:
    tile, col_off, row_off, width, height  = arg[0], int(arg[1]), int(arg[2]), int(arg[3]), int(arg[4])
    ttprint(tile, col_off, row_off)
    window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
    print(window)

    data = read_rasters_cpp(raster_files=raster_files, window=window, gdal_opts=gdal_opts, verbose=False)
    data[0,:] = np.where(data[0] > 0, 2, np.where(data[0] < 0, 1, 0))
    data[1,:] = np.where(np.isnan(data[1]), 0, data[1])
    data[2,:] = np.where(np.isnan(data[2]), 0, data[2])
    
    print('short_veg: ', np.unique(data[0]))
    print('land.cover: ', np.unique(data[1]))
    print('continent.mask: ', np.unique(data[2]))

    d0, d1, d2, counts = unique_counts_bitwise(data)
    
    uniqs  = { Path(r).stem:d for r,d in zip(raster_files,[d0, d1, d2]) }
    uniqs['count'] = counts
    uniqs['tile'] = tile

    uniqs = pd.DataFrame(uniqs)

    pq.write_to_dataset(
      Table.from_pandas(uniqs),
      out_file,
      partition_cols=['tile'],
      compression="snappy",
      version="2.4",
    )

    del data

ttprint("End")

arg_fn = '/mnt/atlas/WRI/args.in'
args = np.loadtxt(arg_fn, dtype=str)