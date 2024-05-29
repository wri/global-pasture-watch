gdal_rasterize -tr 0.000833333333333 0.000833333333333 -burn 1 -add -at -a_nodata 0 -ot Byte -co COMPRESS=DEFLATE -co TILED=YES -co NUM_THREADS=16 -co SPARSE_OK=TRUE -co BIGTIFF=YES highways.gpkg highways.tif
gdal_proximity.py -values 1,2,3,4,5 -ot UInt16 -maxdist 65000 -nodata 65535 -distunits PIXEL -ot Byte -co COMPRESS=DEFLATE -co TILED=YES -co NUM_THREADS=16 -co SPARSE_OK=TRUE -co BIGTIFF=YES highways.tif highways_low_density.tif