from eumap.misc import GoogleSheet
from pathlib import Path 
import rasterio
import math
import re

def replace_year(fn_layer, year = None):
    y, y_m1, y_p1 = ('', '', '')
    if year is not None:
      y, y_m1, y_p1 = str(year), str((year - 1)), str((year + 1))

    fn_layer = str(fn_layer)

    return fn_layer \
        .replace('{year}', y) \
        .replace('{year_minus_1}', y_m1) \
        .replace('{year_plus_1}', y_p1) \

def warp_cmd(r, w, internal_url, year = None):
    #opts = '--config GDAL_CACHEMAX 2048 -co BLOCKSIZE=1024 -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=8 -co LEVEL=9 -of COG'
    opts = '--config GDAL_CACHEMAX 2048 -co COMPRESS=DEFLATE -co TILED=YES -co NUM_THREADS=8 -co SPARSE_OK=TRUE -co BIGTIFF=YES'
    
    tr = re.findall("\d+", r['spatial_resolution'])[0]
    prefix = '_'.join(Path(r['internal_url']).name.split('_')[0:-1])
    suffix = Path(r['internal_url']).name.split('_')[-1]
    out_name = '_'.join([prefix, w['area'], w['crs_id'], suffix])
    if year is not None:
        out_name = replace_year(out_name, int(year))
    
    return f"gdalwarp -overwrite -t_srs '{w['crs']}'" \
        + f" -te {' '.join(w['bounds'])}" \
        + f" -tr {tr} {tr}"\
        + f" -r 'near'"\
        + f" {opts}"\
        + f" /vsicurl/{internal_url}" \
        + f" {out_name}"

key_file = '/mnt/apollo/stac/gaia-319808-913d36b5fca4.json'
url = 'https://docs.google.com/spreadsheets/d/1GI4XW4qenBA2emZsIiHS_jhCY-C7PmDwWe_3RhoWUb4/edit#gid=224784917'

gsheet = GoogleSheet(key_file, url, verbose=False)

warp_params = []

for area in gsheet.MVP_landsat['area'].unique():
    row = gsheet.MVP_landsat[gsheet.MVP_landsat['area'] == area].iloc[0]
    raster_file = row['internal_url']
    raster_file = replace_year(raster_file, int(row['end_year']))
    
    ds = rasterio.open(raster_file)
    
    crs_id = str(Path(raster_file).name).split('_')[-2]
    x = [ds.bounds.left, ds.bounds.right]
    y = [ds.bounds.top, ds.bounds.bottom]
    crs = str(ds.crs)
    if area == 'eumap': # fixing epsg for eumap
        crs = 'EPSG:3035'

    warp_params.append({
        'area': area, 
        'crs': crs, 
        'bounds': [str(b) for b in [min(x), min(y), max(x), max(y)]], 
        'crs_id': crs_id
    })

for w in warp_params:
    for _, r in gsheet.Global.iterrows():
        if r['type'] == 'timeless':
            print(warp_cmd(r, w, r['internal_url']))
        if r['type'] == 'temporal':
            for year in range(2000, 2022):
                yaux = int(year)
                if yaux < int(r['start_year']):
                    yaux = r['start_year']
                elif yaux > int(r['end_year']):
                    yaux = r['end_year']
                internal_url = replace_year(r['internal_url'], int(yaux))

                if internal_url == 'global/dtm/dtm_slope.pct_copernicus.glo.fabdem.dtm_m_30m_s0..0cm_2010..2015_v1.0.tif':
                    internal_url = '/mnt/lacus/raw/FABDEM/terrain_attributes/dtm_slope.pct_copernicus.glo.fabdem.dtm_m_30m_s0..0cm_2010..2015_v1.0.vrt'
                if internal_url == 'global/dtm/dtm_aspect.deg_copernicus.glo.fabdem.dtm_m_30m_s0..0cm_2010..2015_v1.0.tif':
                    internal_url = '/mnt/lacus/raw/FABDEM/terrain_attributes/dtm_aspect.deg_copernicus.glo.fabdem.dtm_m_30m_s0..0cm_2010..2015_v1.0.vrt'

                print(warp_cmd(r, w, internal_url, year))