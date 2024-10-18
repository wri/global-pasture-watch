import geopandas as gpd
tiles = gpd.read_file('ard2_final_status.gpkg')

GDAL_ARGS='-ot UInt16 --config GDAL_CACHEMAX 10240 -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co ZLEVEL=9 -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 -co NUM_THREADS=8 -co SPARSE_OK=TRUE'

for _,row in tiles.iterrows():
    minx, miny, maxx, maxy = row.geometry.bounds
    tile = row['TILE']
    print(f"gdal_translate {GDAL_ARGS} -projwin {minx} {maxy} {maxx} {miny} ./urbanmask/WSF_EVO_DLR/WSFevolution_v1.vrt urbanmask/tiles/WSFEVO_{tile}.tif")
    print(f"gdal_translate {GDAL_ARGS} -projwin {minx} {maxy} {maxx} {miny} ./urbanmask/WSF_EVO_DLR/WSF2019.vrt urbanmask/tiles/WSF2019_{tile}.tif")