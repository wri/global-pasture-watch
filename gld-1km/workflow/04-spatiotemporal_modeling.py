import pandas as pd
import numpy as np

import lightgbm as lgb

from scipy.stats import uniform, randint
from sklearn.experimental import enable_halving_search_cv 
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import make_scorer, d2_tweedie_score
from sklearn.metrics import mean_squared_log_error, mean_tweedie_deviance
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import PowerTransformer

from pathlib import Path
from skmap.misc import ttprint
import joblib

def run_if_not_exists(fn, func):
  if Path(fn).exists():
    ttprint(f'Using variables from {fn}')
    result = joblib.load(fn)
  else:
    result = func() 
    joblib.dump(result, fn, compress='lz4')
  globals().update(result)

def feature_selection(data, covs, target_column, weight_column, 
  n_features_to_select, estimator, seed = 0):
  
  rfe_step = int(len(covs) * 0.05)
  ttprint(f"Finding the best {n_features_to_select} features using RFE (step={rfe_step})")

  rfe = RFE(estimator=estimator, step=rfe_step, n_features_to_select=n_features_to_select, verbose=1)
  rfe.fit(data[covs], data[target_column], **{'sample_weight': data[weight_column]})

  return covs[rfe.support_]

def lgb_hyper_parameter_tunning(data, data_eval, covs, target_column, weight_column,
  lgb_estimator, param_distributions, scoring, eval_metric, cv, seed = 0):
  
  min_resources = int(data.shape[0]*0.05)
  if min_resources <= 5:
    min_resources = int(data.shape[0]*0.5)
    
  max_resources = data.shape[0]
  factor = 1.5

  ttprint(f"Finding the best hyper-parameters using HalvingRandomSearchCV ({min_resources}--{max_resources} factor={factor})")

  hyperpar_lgb = HalvingRandomSearchCV(
      estimator = lgb_estimator(),
      scoring = scoring,
      param_distributions = param_distributions,
      factor = factor,
      verbose = 1,
      min_resources = min_resources,
      max_resources = max_resources,
      cv = cv,
      random_state=seed
  )

  hyperpar_lgb.fit(data[covs],  data[target_column], 
                  groups = data[spatial_cv_column], sample_weight = data[weight_column],
                  eval_set = [(data_eval[covs], data_eval[target_column])],
                  eval_sample_weight = [data_eval[weight_column].to_numpy()],
                  eval_metric = eval_metric)

  print(hyperpar_lgb.best_score_, hyperpar_lgb.best_params_)
  ttprint(f'Best score: {hyperpar_lgb.best_score_}')
  ttprint(f'Best hyper-parameters:\n{hyperpar_lgb.best_params_}')

  best_lgb = lgb_estimator(verbosity = 2, **hyperpar_lgb.best_params_)
  best_lgb.fit(data[covs],  data[target_column], 
          sample_weight = data[weight_column],
          eval_set = [(data_eval[covs], data_eval[target_column])],
          eval_sample_weight = [data_eval[weight_column].to_numpy()],
          eval_metric = eval_metric,)

  best_params = hyperpar_lgb.best_params_ 
  best_params['n_estimators'] = best_lgb.booster_.current_iteration()

  ttprint(f'Optimized n_estimators: {best_params["n_estimators"]}')

  del best_params['early_stopping_min_delta']
  del best_params['early_stopping_rounds']
  del best_params['metric']
  
  return best_params

def rf_hyper_parameter_tunning(data, covs, target_column, target_suff, weight_column,
  rf_estimator, param_distributions, cv, seed = 0):
  
  min_resources = int(data.shape[0]*0.05)
  if min_resources <= 5:
    min_resources = int(data.shape[0]*0.5)
    
  max_resources = data.shape[0]
  factor = 1.5

  ttprint(f"Finding the best hyper-parameters using HalvingRandomSearchCV ({min_resources}--{max_resources} factor={factor})")

  hyperpar_rf = HalvingRandomSearchCV(
      estimator = rf_estimator(),
      scoring = scoring[target_suff],
      param_distributions = param_distributions,
      factor = factor,
      verbose = 1,
      min_resources = min_resources,
      max_resources = max_resources,
      cv = cv,
      random_state=seed
  )

  hyperpar_rf.fit(data[covs],  data[target_column], 
                  groups = data[spatial_cv_column], sample_weight = data[weight_column])

  best_params = hyperpar_rf.best_params_ 
  ttprint(f'Optimized rf params: {best_params}')
  
  return best_params

def rmsle(y_true, y_pred, sample_weight):
    return "rmsle", mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight, squared=False), False

def mtdp2(y_true, y_pred, sample_weight):
    return "mtdp2", mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2), False

def mtdp1(y_true, y_pred, sample_weight):
    return "mtdp1", mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1), False

def d2p1(y_true, y_pred, sample_weight):
    return "d2p1", d2_tweedie_score(y_true, y_pred, sample_weight=sample_weight, power=1), True

wd = '/mnt/tupi/WRI/livestock_global_modeling/livestock_census_ard'
sample_fn = f'{wd}/gpw_livestock.animals_gpw.fao.faostat.malek.2024_zonal.samples_20000101_20231231_go_epsg.4326_v1.pq'
#sample_fn = f'{wd}/subsample.pq'
model_dir = f'{wd}/zonal_models_zeros_nowei_prod_v20250924_cur/'

Path(model_dir).mkdir(parents=True, exist_ok=True)

ttprint(f"Loading {sample_fn}")
data = pd.read_parquet(f'{sample_fn}')

animal_types = ['cattle', 'sheep', 'goat', 'horse', 'buffalo']
target_cols = ['density', 'density_boxcox']
#    'goat': 1124, #311,
#    'horse': 94, #52,
#    'buffalo': 797 #338,
#}

max_density = {
    'cattle': 1511, #1428,
    'sheep': 713, #534,
    'goat': 832, #311,
    'horse': 83, #52,
    'buffalo': 490 #338,
}

adhoc_mask = np.logical_not(
    np.logical_or.reduce([
        np.logical_and.reduce([
            data['gazName'].str.contains('Russian Federation'),
            data['area_km2'] <= 10
        ]),
        data['gazName'] == 'United States of America.Alaska.North Slope'
    ])
)
data = data[adhoc_mask]

spatial_cv_column = 'gazName'
weight_column = 'weight'
cv_njobs = 5
cv_folds = cv_njobs
seed = 1989

# Weigth
data[weight_column] = 1
feat_to_sel = 50

rfe_criterion = {
    'density': 'poisson',
    'density_boxcox': 'squared_error',
}

rf_estimator = RandomForestRegressor
rf_param_distributions = {
  'density': {
    "criterion": ["poisson"],
    "max_depth": randint(10, 200),
    "max_features": uniform(loc=0, scale=1),
    "min_samples_split": randint(2, 60),
    "min_samples_leaf": randint(3, 50),
    "bootstrap": [True],
    "n_jobs": [-1],
    "n_estimators": randint(60, 200),
  },
  'density_boxcox': {
    "criterion": ["squared_error"],
    "max_depth": randint(10, 200),
    "max_features": uniform(loc=0, scale=1),
    "min_samples_split": randint(2, 60),
    "min_samples_leaf": randint(3, 50),
    "bootstrap": [True],
    "n_jobs": [-1],
    "n_estimators": randint(60, 200),
  }
}

pct_early_stopping = 0.50
lgb_estimator = lgb.LGBMRegressor
lgb_param_distributions = {
    'density': {
      #"num_leaves" : randint(100, 1000),
      'learning_rate': uniform(loc=0.01, scale=0.09),
      'n_estimators': [1000],
      #'objective': ['poisson'],
      'objective': ['poisson'],
      #'lambda_l1': uniform(loc=10, scale=1000),
      #'lambda_l2': uniform(loc=10, scale=10000),
      #'min_data_in_leaf': randint(10, 100),
      'min_data_in_leaf': randint(5, 50),
      'poisson_max_delta_step': uniform(loc=0.7, scale=0.2),
      #'tweedie_variance_power': list(np.arange(1,2,0.025)), #uniform(loc=1.1, scale=0.8),
      'metric': ['rmse'],
      'early_stopping_rounds' : [20],
      'early_stopping_min_delta': [0.01],
      'data_sample_strategy': ['bagging', 'goss'],
      'max_bin': [512, 1024],
      'num_threads': [-1],
      #'feature_fraction': list(np.arange(0.2,0.8,0.1)),
      #'extra_trees': [False, True],
      #'bagging_fraction': list(np.arange(0.2,0.8,0.1)),
      #'bagging_freq': randint(1, 5),
      'verbose': [-1]
    },
    'density_boxcox': {
        #"num_leaves" : randint(20, 200),
        'learning_rate': uniform(loc=0.05, scale=0.50),
        'n_estimators': [1000],
        'objective': ['regression'],
        #'lambda_l1': uniform(loc=0.1, scale=10),
        #'lambda_l2': uniform(loc=0.1, scale=25),
        'min_data_in_leaf': randint(5, 50),
        'metric': ['rmse'],
        'early_stopping_rounds' : [20],
        'early_stopping_min_delta': [0.01],
        'data_sample_strategy': ['bagging'],
        'max_bin': [512, 1024],
        'num_threads': [-1],
        #'feature_fraction': list(np.arange(0.2,0.8,0.1)),
        #'extra_trees': [False, True],
        #'bagging_fraction': list(np.arange(0.2,0.8,0.1)),
        #'bagging_freq': randint(1, 5),
        'verbose': [-1]
    }
}

scoring = {
    'density': make_scorer(d2_tweedie_score, power = 1, greater_is_better = True),
    'density_boxcox': 'r2',
}

eval_metric = {
    'density': d2p1,
    'density_boxcox': 'r2',
}

ml_args = [ (animal_type, f'{animal_type}_{target_column}', target_column) for animal_type in animal_types for target_column in target_cols ]

for (animal_type, target_column, target_suff) in ml_args:
  ttprint(f"Modeling {target_column} for {animal_type}")
  seed += 1 # Changing seed across each model
  
  density_col = f'{animal_type}_density'
  model_fn = f'{model_dir}/{animal_type}.{target_column}'

  data_ani = data[
    np.logical_and.reduce([
      data[density_col] <= max_density[animal_type],
      data[f'ind_{animal_type}'] == 1
    ])
  ].copy()
  
  cov_idx = data_ani.columns.get_loc(list(data_ani.columns[data_ani.columns.str.contains('ind_')])[-1]) + 1
  #covs = data_ani.columns[cov_idx:]
  #covs = data_ani.columns[cov_idx:].drop([
  #  'wilderness_li2022.human.footprint_p_1km_s_year0101_year1231_go_epsg.4326_v16022022'
  #])
  # Produce artifacts on high latitudes
  covs = data_ani.columns[cov_idx:].drop([
    'lcv_accessibility.to.ports_map.ox.var1_m_1km_s0..0cm_2015_v14052019',
    'lcv_accessibility.to.ports_map.ox.var2_m_1km_s0..0cm_2015_v14052019',
    'lcv_accessibility.to.ports_map.ox.var3_m_1km_s0..0cm_2015_v14052019',
    'lcv_accessibility.to.ports_map.ox.var4_m_1km_s0..0cm_2015_v14052019',
    'lcv_accessibility.to.ports_map.ox.var5_m_1km_s0..0cm_2015_v14052019'
  ])
  #print(list(covs))

  target_pt = None
  if target_suff == 'density_boxcox':
    target_pt = PowerTransformer(method='box-cox')
    data_ani[target_column] = target_pt.fit_transform(data_ani[density_col].to_numpy().reshape(-1,1))

  data_ani_calib = data_ani[data_ani[f'{animal_type}_ml_type'] == 'calibration']
  data_ani_train = data_ani[data_ani[f'{animal_type}_ml_type'] == 'training']
  data_all = pd.concat([data_ani_train, data_ani_calib])
  
  ttprint(f"Training set: {data_ani_train[covs].shape}")
  ttprint(f"Calibration set: {data_ani_calib[covs].shape}")

##################################################
### Feature Selection
##################################################
  def run():
    
    return {
      'covs_rfe': feature_selection(
        data = data_ani_calib, 
        covs = covs, 
        target_column = target_column, 
        weight_column = weight_column, 
        n_features_to_select = feat_to_sel,
        estimator = RandomForestRegressor(n_jobs=50, criterion=rfe_criterion[target_suff], random_state=seed),
        seed = seed
      )
    }
  
  run_if_not_exists(f'{model_fn}_rfecv.lz4', run)

##################################################
### Hyper-parameter tunning
##################################################

  def run():
    
    cv_blocks = data_all[spatial_cv_column].unique()
    eval_block = np.random.choice(cv_blocks, int(cv_blocks.shape[0] * pct_early_stopping))
    data_calib_ani_eval_lgb = data_all[data_all[spatial_cv_column].isin(eval_block)]
    data_calib_ani_lgb = data_all.drop(index=data_calib_ani_eval_lgb.index)

    return {
      'data_calib_ani_lgb': data_calib_ani_lgb,
      'data_calib_ani_eval_lgb': data_calib_ani_eval_lgb,
      'best_params': lgb_hyper_parameter_tunning(
        data = data_calib_ani_lgb,
        data_eval = data_calib_ani_eval_lgb,
        covs = covs_rfe,
        target_column = target_column,
        weight_column = weight_column,
        lgb_estimator = lgb_estimator,
        param_distributions = lgb_param_distributions[target_suff],
        scoring = scoring[target_suff],
        eval_metric = eval_metric[target_suff],
        cv = GroupKFold(cv_folds),
        seed = seed
      )
    }
  
  run_if_not_exists(f'{model_fn}.lgb_hyperparams.lz4', run)

##################################################
### CV
##################################################

  def run():

    pred_cv = cross_val_predict(
        lgb_estimator(**best_params), 
        data_ani_train[covs_rfe], 
        data_ani_train[target_column], 
        n_jobs = 2, 
        groups = data_ani_train[spatial_cv_column], 
        verbose = True,
        cv = GroupKFold(cv_folds),
        fit_params={
            'sample_weight': data_ani_train[weight_column]
        }
    )

    return {
      'df_cv': pd.DataFrame({
        'predicted': pred_cv,
        'expected':  data_ani_train[target_column],
        'cv_group': data_ani_train[spatial_cv_column],
        'weight': data_ani_train[weight_column]
      })
    }

  #run_if_not_exists(f'{model_fn}.lgb_cv.lz4', run)

##################################################
### Final model
##################################################
  
  def run():

    prod_lgb = lgb.LGBMRegressor(**best_params)
    prod_lgb.fit(data_all[covs_rfe],  data_all[target_column], 
                    sample_weight=data_all[weight_column])

    return {
      'prod_lgb': prod_lgb,
      'target_pt': target_pt
    }

  run_if_not_exists(f'{model_fn}.lgb_prod.lz4', run)

##################################################
### RF Hyper-parameter tunning
##################################################
    
  def run():

    return {
      'best_params': rf_hyper_parameter_tunning(
        data = data_ani_calib,
        covs = covs_rfe,
        target_column = target_column,
        target_suff = target_suff,
        weight_column = weight_column,
        rf_estimator = rf_estimator,
        param_distributions = rf_param_distributions[target_suff],
        cv = GroupKFold(cv_folds),
        seed = seed
      )
    }
  
  run_if_not_exists(f'{model_fn}.rf_hyperparams.lz4', run)
    
##################################################
### RF CV
##################################################

  def run():

    pred_cv = cross_val_predict(
        rf_estimator(**best_params), 
        data_ani_train[covs_rfe], 
        data_ani_train[target_column], 
        n_jobs = 2, 
        groups = data_ani_train[spatial_cv_column], 
        verbose = True,
        cv = GroupKFold(cv_folds),
        fit_params={
            'sample_weight': data_ani_train[weight_column]
        }
    )

    return {
      'df_cv': pd.DataFrame({
        'predicted': pred_cv,
        'expected':  data_ani_train[target_column],
        'cv_group': data_ani_train[spatial_cv_column],
        'weight': data_ani_train[weight_column]
      })
    }

  #run_if_not_exists(f'{model_fn}.rf_cv.lz4', run)

##################################################
### RF Final model
##################################################
  
  def run():

    prod_lgb = rf_estimator(**best_params)
    prod_lgb.fit(data_all[covs_rfe],  data_all[target_column], 
                    sample_weight=data_all[weight_column])

    return {
      'prod_rf': prod_lgb,
      'target_pt': target_pt
    }

  run_if_not_exists(f'{model_fn}.rf_prod.lz4', run)
    
##################################################
### Final model
##################################################
    
  ttprint("Cleaning previous execution state")
  del covs_rfe, data_calib_ani_lgb, data_calib_ani_eval_lgb, best_params, prod_lgb, target_pt