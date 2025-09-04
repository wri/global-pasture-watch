from pathlib import Path
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from skmap.misc import ttprint
import bottleneck as bn
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skmap_bindings as skb
import numpy as np
import math

from sklearn.metrics import r2_score, d2_tweedie_score
from sklearn.metrics import mean_squared_error #root_mean_squared_error
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
from matplotlib import colors

def gdalwarp_cmds(df, indir, outdir, end_year, te, tr, t_srs):
    cmds = []

    indir = 'raster_layers'
    outdir = 'datacube'
    end_year = 2022

    for r_method, rows in df.groupby('r_method'):
        for _, row in rows.iterrows():

            layername = row['layername']
            if row['type'] == 'static':
                outvrt = f'{outdir}/{layername}.vrt'
                if not Path(outvrt).exists():
                    cmds.append(
                        f"gdalwarp -te {te} -t_srs '{t_srs}' -r {r_method} -tr {tr} {indir}/{layername}.tif {outvrt}"
                    )
            elif row['type'] == 'temporal':
                for y in range(int(row['start_year']), end_year + 1):
                    ly = y
                    if ly > int(row['end_year']):
                        ly = int(row['end_year'])
                    y_basename_in = layername.replace('{year}', str(ly))
                    y_basename_out = layername.replace('{year}', str(y))
                    outvrt = f'{outdir}/{y_basename_out}.vrt'
                    if not Path(outvrt).exists():
                        cmds.append(
                            f"gdalwarp -te {te} -t_srs '{t_srs}' -r {r_method} -tr {tr} {indir}/{y_basename_in}.tif {outvrt}"
                        )

    return cmds

def extract_years(polygon_samp):
    cols = polygon_samp.columns[polygon_samp.columns.str.contains('_2')]
    cols = cols[(polygon_samp[cols] >= 0).to_numpy().flatten()]
    years = cols.map(lambda x: int(x.split('_')[1])).unique()
    years = years.drop(2023, errors='ignore')
    years = years.drop(2024, errors='ignore')
    return sorted(years)

def extract_window(polygon_samp, datacube_df, crs):
    raster_layers = list(datacube_df['path'])
    minx, miny, maxx, maxy = polygon_samp.to_crs(crs).total_bounds
    return  from_bounds(minx, miny, maxx, maxy, rasterio.open(raster_layers[0]).transform).round_lengths().round_offsets()

def gen_covs_info(static_covs, temporal_covs, years, base_path):
    
    path, rtype = [], []
    if static_covs is not None:
        path += [ f'{base_path}/{c}.vrt' for c in static_covs ]
        rtype += [ 'static' for c in static_covs ]
    
    for y in years:
        path += [ f"{base_path}/{c.replace('{year}', str(y))}.vrt" for c in temporal_covs ]
        rtype += [ y for c in temporal_covs ]
    
    return pd.DataFrame({
        'path': path,
        'type': rtype
    })

def gen_mask_info(years, base_path):
    
    temporal_covs = [
        'gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{year}0101_{year}1231_go_epsg.4326_v1'
    ]
    
    return gen_covs_info(None, temporal_covs, years, base_path)

def gen_polygon_mask(polygon_samp, mask_info, crs):
    window = extract_window(polygon_samp, mask_info, crs)
    ds = rasterio.open(mask_info['path'][0])
    transform = (
      ds.transform[0], 0.0,  ds.transform[2] + window.col_off * ds.transform[0],
      0.0, ds.transform[4],  ds.transform[5] + window.row_off * ds.transform[4]
    )
    out_shape = (window.height,window.width)
    polygon_mask_temp = rasterize(polygon_samp.to_crs(crs).geometry, fill=0, out_shape=out_shape, transform=transform, dtype='float32')
    polygon_mask = np.empty((window.height * window.width,), dtype='float32')
    polygon_mask[:] = polygon_mask_temp.flatten()
    return polygon_mask

def spationtemporal_masked_zonal_mean(polygon_mask, covs_data, covs_info, mask_data, mask_info, mask_ignore_val, n_jobs = 4, verbose=False):
    masked_zonal = {}

    for year in mask_info['type']:

        covs_info_y = covs_info[np.logical_or.reduce([
          covs_info['type'] == year,
          covs_info['type'] == 'static',
        ])].copy()

        covs_info_y.loc[:,'label'] = covs_info_y['path'].apply(lambda f: Path(f).stem)
        covs_info_y.loc[covs_info_y['type'] != 'static','label'] = covs_info_y[covs_info_y['type'] != 'static']['label'].apply(lambda f: f.replace(f'_{year}','_year').replace(f'..{year}','..year'))

        covs_idx = list(covs_info_y.index)
        covs_label = list(covs_info_y['label'])
        covs_label += [ 'mask_sum' ]

        row_mask = mask_info[mask_info['type'] == year].iloc[0]
        idx = row_mask.name
        key = year
        n_covs = len(covs_idx)

        mask = np.logical_and(
            (mask_data[idx,:] != mask_ignore_val), 
            (polygon_mask == 1)
        ).astype('float32')

        if verbose:
            ttprint(f"Calculating zonal mean for {n_covs} layers and zonal sum for 1 mask on {year}")
        
        covs_data_idx = np.empty((n_covs, covs_data.shape[1]), dtype='float32')
        covs_data_idx[:] = covs_data[covs_idx,:]
        skb.maskDataRows(covs_data_idx, n_jobs, range(0,n_covs), mask, 0, np.nan)
        mean_data = np.empty((len(covs_idx)), dtype='float32')
        skb.nanMean(covs_data_idx, n_jobs, mean_data)
        mean_data = np.round(mean_data)

        land_data = np.empty((1, covs_data.shape[1]), dtype='float32')
        land_data[:] = mask_data[idx,:]
        skb.maskDataRows(land_data, n_jobs, [0], mask, 0, np.nan)
        sum_land = bn.nansum(land_data)
        mean_data = np.append(mean_data, sum_land)

        masked_zonal[key] = np.round(mean_data)

    zonal_df = pd.DataFrame(masked_zonal.values(), columns=covs_label)
    zonal_df['year'] = masked_zonal.keys()
    
    return zonal_df

def merge_livestock_data(polygon_samp, polygon_zonal, animals, area_col, additional_cols):
    
    polygon_zonal['polygon_idx'] = polygon_samp.index[0]
    covs_name = polygon_zonal.columns.drop([area_col,'year','polygon_idx'])
    
    polygon_zonal = polygon_zonal.set_index('polygon_idx', drop=True).merge(
        polygon_samp,
        left_index = True,
        right_index = True
    )
    
    result = []
    
    for year, rows in polygon_zonal.groupby('year'):
        
        row_cols = additional_cols + [area_col, 'year','geometry']

        for animal in animals:
            density_col = f'{animal}_density'
            rows[density_col] = rows[f'{animal}_{year}'] / rows[area_col]
            row_cols.append(density_col)

            heads_col = f'{animal}_heads'
            rows[heads_col] = rows[f'{animal}_{year}']
            row_cols.append(heads_col)

            rows.loc[np.isinf(rows[f'{animal}_density']),f'{animal}_density'] = np.nan
            rows.loc[rows[f'{animal}_density'] == 0,f'{animal}_density'] = np.nan

        rows['geometry'] = gpd.GeoSeries(rows['geometry'])
        rows['year'] = year
        result.append(rows[row_cols + list(covs_name)])

    return pd.concat(result).reset_index(drop=True)

def concordance_correlation_coefficient(y_true, y_pred, wei):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = DescrStatsW(df.to_numpy(), weights=wei).corrcoef[0][1]
    # Means
    mean_true = np.average(y_true, weights=wei)
    mean_pred = np.average(y_pred, weights=wei)
    # Population variances
    #var_true = np.var(y_true)
    var_true = DescrStatsW(y_true, weights=wei).var
    #print(var_true, var_true1)
    #var_pred = np.var(y_pred)
    var_pred = DescrStatsW(y_pred, weights=wei).var
    # Population standard deviations
    sd_true = DescrStatsW(y_true, weights=wei).std
    sd_pred = DescrStatsW(y_pred, weights=wei).std
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator

def histogram_plot(preds,name):
    fig, ax = plt.subplots(1)
    preds[['true','pred']].rename(columns={'true': 'Observed', 'pred':'Predicted'}).plot.hist(
        bins=128, histtype='stepfilled', alpha=0.4, linewidth=1.5, log=False, xlabel='Grain yield (kg/ha)',
        title=f"Model: {name}\nDistribution of observed & predicted",ax=ax
    )
    plt.show()
    
def density_plot(df, name, expected='true', predicted='pred', title_base='Mayze yield based on 5-fold spatial CV', 
                 unit='', figsize=(9,7), cmap="inferno_r"):

    n_samples = df.shape[0]
    df.loc[df['true'] < 0,'true'] = 0.000000001
    df.loc[df['pred'] < 0,'pred'] = 0.000000001

    r2 = r2_score(df[expected], df[predicted])
    d2 = d2_tweedie_score(df[expected], df[predicted], power=1)
    try:
        rmse = root_mean_squared_error(df[expected], df[predicted])
    except:
        rmse = mean_squared_error(df[expected], df[predicted], squared=False)
        
    ccc = concordance_correlation_coefficient(df[expected], df[predicted], df['weight'])
    stats=f'R2={r2:.3f}\nD2={d2:.3f}\nRMSE={rmse:.0f}\nCCC={ccc:.3f}'

    max_val = np.max(df[[predicted,expected]].to_numpy().flatten()) 
    
    if unit != '':
        unit = f'({unit})'
    
    exp_lbl = f'Observed {unit}'
    pre_lbl = f'Predicted {unit}'
    remap_lbls = { expected: exp_lbl }
    remap_lbls[predicted] = pre_lbl

    plot = df.rename(columns=remap_lbls).plot.hexbin(x=exp_lbl, y=pre_lbl, mincnt=1, gridsize=32,  cmap=cmap, extent=[-0.2,max_val,-0.2,max_val],
                            title=f'Model: {name} \n{title_base} ({n_samples:,} samples)', bins='log', figsize=figsize)
    plot.axline([0, 0], [1, 1], color='silver')

    plot.text(0.03, 0.97, stats, transform=plot.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='square,pad=.6',
        facecolor=colors.to_rgba('white', alpha=0.8)))
    fig = plot.get_figure()
    
def importance_plot(feature_cols, importance, title, top_n = 30, figsize=(3,7), color = 'blue', model='rf', remap_covs={},save=False):
    var_imp = pd.DataFrame({'name':feature_cols, 'importance': importance})
    var_imp.index = var_imp['name']
    ax = var_imp.sort_values('importance', ascending=False)[0:top_n].sort_values('importance').plot(kind = 'barh', 
            figsize=figsize, title = title, color = color, legend=False)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Features")
