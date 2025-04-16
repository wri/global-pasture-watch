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
from eumap.misc import ttprint
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
  
  min_resources = int(data.shape[0]*0.05) #500
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
                  eval_sample_weight = [(data_eval[weight_column])],
                  eval_metric = eval_metric)

  print(hyperpar_lgb.best_score_, hyperpar_lgb.best_params_)
  ttprint(f'Best score: {hyperpar_lgb.best_score_}')
  ttprint(f'Best hyper-parameters:\n{hyperpar_lgb.best_params_}')

  best_lgb = lgb_estimator(verbosity = 2, **hyperpar_lgb.best_params_)
  best_lgb.fit(data[covs],  data[target_column], 
          sample_weight = data[weight_column],
          eval_set = [(data_eval[covs], data_eval[target_column])],
          eval_sample_weight = [(data_eval[weight_column])],
          eval_metric = eval_metric)

  best_params = hyperpar_lgb.best_params_ 
  best_params['n_estimators'] = best_lgb.booster_.current_iteration()

  ttprint(f'Optimized n_estimators: {best_params["n_estimators"]}')

  del best_params['early_stopping_min_delta']
  del best_params['early_stopping_rounds']
  del best_params['metric']
  
  return best_params

def rmsle(y_true, y_pred, sample_weight):
    return "rmsle", mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight, squared=False), False

def mtdp2(y_true, y_pred, sample_weight):
    return "mtdp2", mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2), False

def mtdp1(y_true, y_pred, sample_weight):
    return "mtdp1", mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1), False

def d2p1(y_true, y_pred, sample_weight):
    return "d2p1", d2_tweedie_score(y_true, y_pred, sample_weight=sample_weight, power=1), True
  
wd = '/mnt/tupi/WRI/short_vegetation_height/short_veg_ard'
model_dir = f'{wd}/models_multi_modis_lc_bootstrap'

Path(model_dir).mkdir(parents=True, exist_ok=True)
target_cols = [ 'med_ht' ]
#filter_col = 'p95_ht'

#max_density, min_density = 600, 0 #animals/km2
batches = [ 'b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b09', 'b10' ]
#max_val = {
#  'med_ht': 30,
#  'iqr_ht': 30,
#  'p90_ht': 80
#}

weight_cols = {
  'med_ht': 'weight',
  'p05_ht': 'weight',
  'p95_ht': 'weight' 
}

spatial_cv_column = 'glad_tile_id'
covs_start_idx = 53
#weight_column = 'weight'
cv_njobs = 5
cv_folds = cv_njobs
seed = 1989


## Increasing the weight of samples with more grassland proportion
#ttprint("Powering by 2 the weights")
#data['weight'] = np.power(data['weight'],2)
#data.loc[np.isnan(data['weight']),'weight'] = 0
#data['1_weight'] = (1 - data['weight'])
#data.loc[(data['1_weight'] == 0),['1_weight']] = 1

ref_n_features = 40
rfe_criterion = {
    'med_ht': 'poisson',
}

rf_estimator = RandomForestRegressor
rf_param_distributions = {
    "criterion": ["poisson"],
    "max_depth": randint(10, 200),
    "max_features": uniform(loc=0, scale=1),
    "min_samples_split": randint(2, 60),
    "min_samples_leaf": randint(3, 50),
    "bootstrap": [True],
    "n_jobs": [-1],
    "n_estimators": randint(30, 100),
    #'min_impurity_decrease': uniform(loc=0, scale=0.1),
    #'min_weight_fraction_leaf': uniform(loc=0, scale=0.01),
    #'ccp_alpha':uniform(loc=0, scale=0.2)
}

pct_early_stopping = 0.20
lgb_estimator = lgb.LGBMRegressor

scoring = {
    'veg':  make_scorer(d2_tweedie_score, power = 1, greater_is_better = True), #'neg_mean_gamma_deviance' #make_scorer(mean_tweedie_deviance, power = 2, greater_is_better = False),
}

eval_metric = {
    'veg': d2p1,
}

ml_args = [ (target_column) for target_column in target_cols ]

#for animal_type, target_column in zip(animal_types,target_cols):
for (target_column) in ml_args:
  sample_fn = f'{wd}/gpw_short.veg.height.{target_column}_icesat.atl08_point.samples.10.batches_20190101_20221231_go_epsg.4326_v1.pq'
  ttprint(f"Loading {sample_fn}")
  data = pd.read_parquet(f'{sample_fn}')
  data['weight'] = 1
  
  covs = data.columns[covs_start_idx:]
  ttprint("Full list of input covs:")
  ttprint(list(covs))
  
  ttprint(f"Modeling {target_column} for vegetation height")
  seed += 1
  weight_column = weight_cols[target_column]
  
  data_filt = data
  #data_filt = data[
  #  np.logical_and.reduce([
  #    data[filter_col] <= max_val[target_column],
  #    data[weight_column] > 0
  #  ])
  #]

  data_filt_calib = data_filt[data_filt['ml_type'] == 'calibration']
  data_filt_train = data_filt[data_filt['ml_type'] == 'training']
  
  ttprint(f"RFE - Training set: {data_filt_train[covs].shape}")
  ttprint(f"RFE - Calibration set: {data_filt_calib[covs].shape}")

##################################################
### Feature Selection
##################################################
  def run():
    
    return {
      'covs_rfe': feature_selection(
        data = data_filt_calib, 
        covs = covs, 
        target_column = target_column, 
        weight_column = weight_column, 
        n_features_to_select = ref_n_features,
        estimator = RandomForestRegressor(n_jobs=50, criterion=rfe_criterion[target_column], random_state=seed),
        seed = seed
      )
    }
  
  run_if_not_exists(f'{model_dir}/veg.height.{target_column}_rfecv.lz4', run)
  print(covs_rfe)
  
  for b in batches:
    
    b = f'{b}_dtm'
    
    seed += 1 # Changing seed across each batch
    np.random.seed(seed)
    lgb_param_distributions = {
      'veg': {
        "num_leaves" : randint(50, 200),
        'learning_rate': uniform(loc=0.05, scale=0.95),
        'n_estimators': [1000],
        'objective': ['poisson'],
        #'objective': ['poisson', 'gamma', 'tweedie'],
        'lambda_l1': uniform(loc=1, scale=100),
        'lambda_l2': uniform(loc=100, scale=10000),
        'min_data_in_leaf': randint(10, 50),
        'poisson_max_delta_step': uniform(loc=0, scale=2),
        #'tweedie_variance_power': list(np.arange(1,2,0.025)), #uniform(loc=1.1, scale=0.8),
        'metric': ['rmse'],
        'early_stopping_rounds' : [20],
        'early_stopping_min_delta': [0.01],
        'data_sample_strategy': ['bagging'],
        'max_bin': [512],
        'num_threads': [-1],
        #'feature_fraction': list(np.arange(0.2,0.8,0.1)),
        #'extra_trees': [False, True],
        #'bagging_fraction': list(np.arange(0.2,0.8,0.1)),
        #'bagging_freq': randint(1, 5),
        'verbose': [-1]
      }
    }
    
    model_fn = f'{model_dir}/veg.height.{target_column}.{b}'

    data_filt = data.sample(frac=1., replace=True)
    #data_filt = data[
      #np.logical_and.reduce([
        #data[filter_col] <= max_val[target_column],
        #data['batch'].isin([b,'glance']),
        #data[weight_column] > 0
      #])
    #]

    data_filt_calib = data_filt[data_filt['ml_type'] == 'calibration']
    data_filt_train = data_filt[data_filt['ml_type'] == 'training']
    
    ttprint(f"Batch {b} - Training set: {data_filt_train[covs].shape}")
    ttprint(f"Batch {b} - Calibration set: {data_filt_calib[covs].shape}")

    ##################################################
    ### Hyper-parameter tunning
    ##################################################
    def run():
      
      cv_blocks = data_filt_calib[spatial_cv_column].unique()
      eval_block = np.random.choice(cv_blocks, int(cv_blocks.shape[0] * pct_early_stopping))
      data_calib_ani_eval_lgb = data_filt_calib[data_filt_calib[spatial_cv_column].isin(eval_block)]
      data_calib_ani_lgb = data_filt_calib.drop(index=data_calib_ani_eval_lgb.index)

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
          param_distributions = lgb_param_distributions['veg'],
          scoring = scoring['veg'],
          eval_metric = eval_metric['veg'],
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
          data_filt_train[covs_rfe], 
          data_filt_train[target_column], 
          n_jobs = 2, 
          groups = data_filt_train[spatial_cv_column], 
          verbose = True,
          cv = GroupKFold(cv_folds),
          fit_params={
              'sample_weight': data_filt_train[weight_column]
          }
      )

      return {
        'df_cv': pd.DataFrame({
          'predicted': pred_cv,
          'expected':  data_filt_train[target_column],
          'cv_group': data_filt_train[spatial_cv_column],
          'weight': data_filt_train[weight_column]
        })
      }

    #run_if_not_exists(f'{model_fn}.lgb_cv.lz4', run)

  ##################################################
  ### Final model
  ##################################################
    
    def run():

      data_all = pd.concat([data_filt_train, data_filt_calib])
      prod_lgb = lgb.LGBMRegressor(**best_params)
      prod_lgb.fit(data_all[covs_rfe],  data_all[target_column], 
                      sample_weight=data_all[weight_column])

      return {
        'prod_lgb': prod_lgb
      }

    run_if_not_exists(f'{model_fn}.lgb_prod.lz4', run)
      
  ##################################################
  ### Final model
  ##################################################
      
    ttprint(f"Batch {b} - Cleaning previous execution state")
    del data_calib_ani_lgb, data_calib_ani_eval_lgb, best_params, prod_lgb