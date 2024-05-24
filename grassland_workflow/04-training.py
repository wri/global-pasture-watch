from eumap.misc import ttprint
from pathlib import Path
from scipy.signal import argrelmin
from scipy.stats import uniform, randint
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from skmap.mapper import build_ann
from skmap.mapper import LandMapper
from skorch import NeuralNetClassifier, NeuralNet
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from torch import nn, optim
import joblib
import multiprocessing
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
import xgboost as xgb

def run_feature_selection(data, covs, target_column, weight_column, seed, spatial_cv_column = None,
                          subsample_pct = 0.1, n_rep = 5, n_cv = 5, ntrees = 50, local_min_pos = 0, order=2):

    #def log_loss_scorer(clf, X, y_true):
    #    class_labels = clf.classes_
    #    y_pred_proba = clf.predict_proba(X)
    #    error = log_loss(y_true, y_pred_proba, labels=class_labels)
    #    return error
    
    data_sub = data#.sample(int(data.shape[0] * subsample_pct), random_state=seed)
    rfecv_step = int(len(covs) * 0.05)
    rfe_step = int(rfecv_step / 2)
    
    cv, groups = KFold(n_cv), None
    if spatial_cv_column is not None:
        cv, groups = GroupKFold(n_cv), data[spatial_cv_column]
    
    ncpu = multiprocessing.cpu_count()
    if ntrees < ncpu:
        ncpu = ntrees

    ttprint(f"Finding n_features_to_select using RFECV (repetitions={n_rep} step={rfecv_step}, n_samples={data.shape[0]})")
    
    grid_scores = []
    for i in range(0,n_rep):
        rfecv = RFECV(estimator=RandomForestClassifier(ntrees, n_jobs=ncpu, random_state=i), cv=cv, step=rfecv_step, 
                      min_features_to_select=10, n_jobs=n_cv, scoring='neg_log_loss')
        rfecv.fit(data[covs], data[target_column], groups=groups)
        grid_scores += [rfecv.cv_results_['mean_test_score']]

    rfecv_mean_score = np.mean(np.stack(grid_scores, axis=0), axis=0)
    grid_scores_std = np.std(np.stack(grid_scores, axis=0), axis=0)
    
    rfecv_n_features_arr = list(range(rfecv.min_features_to_select, len(covs)+rfecv.step, rfecv.step))
    
    local_min_arr = argrelmin(rfecv_mean_score, order=order)[0]
    local_min = local_min_arr[0]
    if len(local_min_arr) > 1:
        local_min = local_min_arr[local_min_pos]
    
    n_features_to_select = rfecv_n_features_arr[local_min]
    n_features_to_select = 100
    
    ttprint(f"Finding best features using RFE (n_features_to_select = {n_features_to_select})")
    
    rfe = RFE(estimator=RandomForestClassifier(ntrees, n_jobs=ncpu, random_state=n_rep), step=rfe_step, n_features_to_select=n_features_to_select, verbose=1)
    rfe.fit(data[covs], data[target_column], **{'sample_weight': data[weight_column]})

    result = covs[rfe.support_]
    
    return result, rfecv_n_features_arr, rfecv_mean_score, grid_scores_std

class ANNModule(nn.Module):
    def __init__(self, n_features, n_classes=3, n_neurons=256, 
                 dropout_rate = 0.35, n_layers=6, activ=nn.ReLU,
                 out_activ = nn.Softmax):
        super().__init__()
        
        
        self.in_layer = nn.ModuleList([
            nn.Linear(n_features, n_neurons),
            activ()
        ])
        
        self.hiddens = []
        for i in range(0, n_layers):
            self.hiddens.append(nn.ModuleList([
                nn.Linear(n_neurons, n_neurons),
                activ(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(n_neurons)
            ]))
        self.hiddens = nn.ModuleList(self.hiddens )
        
        self.out_layer = nn.ModuleList([
            nn.Linear(n_neurons, n_classes),
            out_activ(dim=-1)
        ])

    def forward(self, X, **kwargs):
        
        for n in self.in_layer:
            X = n(X)
        
        for nodes in self.hiddens:
            for n in nodes:
                X = n(X)
        
        for n in self.out_layer:
            X = n(X)
        
        return X

data_calib = pd.read_parquet('global_samples_calibration_v20240210.pq')
data_train = pd.read_parquet('global_samples_train_v20240210.pq')

cov_idx = list(data_train.columns).index('cv_group') + 1
covs = data_train.columns[cov_idx:]
target_column = 'class'
spatial_cv_column = 'cv_group'
weight_column = 'class_pct'
cv_njobs = 5
cv_folds = 5
seed = 1989

#cov_wei_df = pd.read_csv('cov_weigts.csv').set_index('cov')
#feature_weights = list(cov_wei_df.loc[covs,'weig_norm'])

data_calib = data_calib[data_calib[weight_column]==1]
data_train = data_train[data_train[weight_column]==1]
data = pd.concat([data_train, data_calib])

le = LabelEncoder()
data.loc[:,spatial_cv_column] = le.fit_transform(data[spatial_cv_column])

covs = covs[np.logical_or.reduce([covs.str.contains('accessibility'),
covs.str.contains('blue'),
covs.str.contains('bsf'),
covs.str.contains('bsi'),
covs.str.contains('clm'),
covs.str.contains('dtm.bareearth'),
covs.str.contains('evi'),
covs.str.contains('fapar'),
covs.str.contains('green'),
covs.str.contains('ndti'),
covs.str.contains('ndvi'),
covs.str.contains('ndwi'),
covs.str.contains('nir'),
covs.str.contains('nirv'),
covs.str.contains('red'),
covs.str.contains('road.distance_osm.highways.high.density'),
covs.str.contains('road.distance_osm.highways.low.density'),
covs.str.contains('swir1'),
covs.str.contains('swir2'),
covs.str.contains('thermal'),
covs.str.contains('water.distance_glad.interanual.dynamic.classes'),
covs.str.contains('wv_mcd19a2v061')])]

fn_rfcv = 'model_v20240210/rfecv_100.lz4' 
if Path(fn_rfcv).exists():
    ttprint(f'Reusing best features from {fn_rfcv}')
    locals().update(joblib.load(fn_rfcv))
    ttprint(f"Number of features selected: {len(covs_rfe)}")
else:
    covs_rfe, rfecv_n_features_arr, rfecv_mean_score, rfecv_std_score = run_feature_selection(data_calib, covs, target_column, 
                                                                                              weight_column, seed, spatial_cv_column, local_min_pos=0)
    joblib.dump({
        'covs_rfe':covs_rfe, 
        'rfecv_n_features_arr': rfecv_n_features_arr, 
        'rfecv_mean_score': rfecv_mean_score,
        'rfecv_std_score': rfecv_std_score
    }, fn_rfcv, compress='lz4')

#data = data.sample(100000, random_state=1989)
calibration_idx = data_calib.index

ttprint(f"Training the model with {data.shape}")

max_resources_rf = len(calibration_idx)
max_resources_xb = len(calibration_idx)
max_resources_ann = len(calibration_idx)
#max_resources = 200000
#if max_resources_rf < max_resources and data.shape[0] >= max_resources:
#    max_resources_rf = max_resources
#    max_resources_xb = max_resources
#    max_resources_ann = max_resources
#if max_resources_rf < max_resources:
#    max_resources_rf = data.shape[0]
#    max_resources_xb = data.shape[0]
#    max_resources_ann = data.shape[0]

############################################
### Random Forest
############################################

estimator_rf = RandomForestClassifier(n_jobs=-1, n_estimators=60)

param_distributions = {
    "criterion": [ "gini", "entropy"],
    "max_depth": randint(5, 100),
    "max_features": uniform(loc=0, scale=1),
    "min_samples_split": randint(2, 40),
    "min_samples_leaf": randint(1, 10),
    "class_weight": [None, "balanced", "balanced_subsample" ]
}

hyperpar_rf = HalvingRandomSearchCV(
    estimator = estimator_rf,
    scoring = 'neg_log_loss',
    param_distributions = param_distributions,
    factor = 2,
    verbose = 1,
    min_resources = 500,
    max_resources = max_resources_rf,
    cv = GroupKFold(cv_folds),
    random_state=seed
)

# Best: -0.53294 using 
# hyperparams_rf = {
#    'class_weight': None, 'criterion': 'entropy', 'max_depth': 80, 
#    'max_features': 0.12569685800040376, 'min_samples_leaf': 7, 
#    'min_samples_split': 31
# }

############################################
### XGBoost
############################################

estimator_xb = xgb.XGBClassifier(n_jobs=-1, objective='multi:softmax', booster='gbtree', 
    use_label_encoder=False, eval_metric='mlogloss', random_state=seed)

param_distributions = {
    "tree_method": ['hist', 'approx'],
    "grow_policy": ['depthwise', 'lossguide'],
    "alpha": uniform(loc=0, scale=2),
    "reg_alpha": uniform(loc=0, scale=0.2),
    "eta": uniform(loc=0, scale=2),
    "reg_lambda": uniform(loc=0, scale=0.2),
    "gamma": uniform(loc=0, scale=2),
    "subsample": uniform(loc=0.5, scale=0.5),
    "learning_rate": uniform(loc=0, scale=0.2),
    "colsample_bytree": uniform(loc=0, scale=1),
    "colsample_bylevel": uniform(loc=0, scale=1),
    "colsample_bynode": uniform(loc=0, scale=1),
    "max_depth": randint(10, 100),
    "n_estimators": randint(10, 60)
}

hyperpar_xb = HalvingRandomSearchCV(
    estimator = estimator_xb,
    scoring = 'neg_log_loss',
    param_distributions = param_distributions,
    factor = 2,
    verbose = 1,
    min_resources = 500,
    max_resources = max_resources_xb,
    cv = GroupKFold(cv_folds), 
    random_state=seed
)

# Best: -0.50776 using 
# hyperparams_xb = { 
#    'alpha': 1.7838757685283126, 'colsample_bylevel': 0.14161653040605593, 
#    'colsample_bynode': 0.9085425014055984, 'colsample_bytree': 0.7932264472507218, 
#    'eta': 0.5478914460144433, 'gamma': 1.5841963124106253, 'grow_policy': 'lossguide', 
#    'learning_rate': 0.10071430460941982, 'max_depth': 82, 'n_estimators': 51, 
#    'reg_alpha': 0.055707675853048394, 'reg_lambda': 0.08870765254290161, 
#    'subsample': 0.6269723460503966, 'tree_method': 'approx'
#}

############################################
### ANN Scikit-learn
############################################

from sklearn.neural_network import MLPClassifier

net = MLPClassifier(
    max_iter=100, 
    random_state=1989, 
    early_stopping=True, 
    n_iter_no_change=5,
    verbose=False
)

estimator_ann = Pipeline([
    ('scaler', StandardScaler()),
    ('estimator', net),
])

param_distributions = {
    "estimator__hidden_layer_sizes": [ (i,j) for i in range(4,8) for j in range(32,256,32) ],
    "estimator__batch_size": randint(32, 256),
    "estimator__learning_rate_init": uniform(loc=0.0001, scale=0.001),
    "estimator__activation": ['logistic', 'relu'],
    "estimator__alpha": uniform(loc=0.0001, scale=0.00005),
    "estimator__learning_rate": ['constant', 'adaptive'],
    "estimator__beta_1": uniform(loc=0.65, scale=0.30),
    "estimator__beta_2": uniform(loc=0.65, scale=0.30),
    "estimator__epsilon": uniform(loc=1e-8, scale=1e-9),
    "estimator__solver": ['adam']
}

hyperpar_ann = HalvingRandomSearchCV(
    estimator = estimator_ann,
    scoring = 'neg_log_loss',
    param_distributions = param_distributions,
    factor = 2,
    verbose = 1,
    min_resources = 500,
    max_resources = max_resources_ann,
    cv = GroupKFold(cv_folds), 
    random_state=seed
)

############################################
### ANN Pytorch
############################################

#net = NeuralNetClassifier(
#    ANNModule,
#    max_epochs=50,
#    #lr=0.0005,
#    #criterion=FocalLoss(gamma=0.7),
#    criterion=nn.CrossEntropyLoss,
#    optimizer=optim.NAdam,
#    verbose=0,
#    #train_split=predefined_split(test_nn),
#    train_split=None,#ValidSplit(stratified=True),
#    callbacks=[EarlyStopping(patience=3, monitor='train_loss')],
#    # Shuffle training data on each epoch
#    iterator_train__shuffle=True,
#    iterator_train__batch_size=64,
#    iterator_valid__batch_size=1000000
#)
#
#estimator_ann = Pipeline([
#    ('scaler', StandardScaler()),
#    ('estimator', net),
#])
#
#param_distributions = {
#    "estimator__module__n_neurons": randint(32, 256),
#    "estimator__module__n_layers": randint(2, 8),
#    "estimator__module__dropout_rate": uniform(loc=0.15, scale=0.5),
#    "estimator__module__activ": [nn.Sigmoid, nn.ReLU, nn.ELU],
#    "estimator__module__n_features": [len(covs)],
#    "estimator__lr": uniform(loc=0.0001, scale=0.001),
#    #"estimator__iterator_train__batch_size": randint(16, 256)
#    #"estimator__criterion": [FocalLoss(gamma=0.7), nn.CrossEntropyLoss]
#}
#
#hyperpar_ann = HalvingRandomSearchCV(
#    estimator = estimator_ann,
#    scoring = 'neg_log_loss',
#    param_distributions = param_distributions,
#    factor = 2,
#    verbose = 1,
#    min_resources = 500,
#    max_resources = max_resources_ann,
#    cv = GroupKFold(cv_folds), 
#    random_state=seed
#)

############################################
### Meta-learner
############################################

meta_estimator = LogisticRegression(
    solver='saga', 
    multi_class='multinomial', 
    n_jobs=-1, 
    verbose=True,
    random_state=seed)

# LandMapper
cv_method = GroupKFold(cv_folds)
estimator_list = [estimator_rf, estimator_xb, estimator_ann]
hyperpar_selection_list = [hyperpar_rf, hyperpar_xb, hyperpar_ann]

#estimator_list = [estimator_rf, estimator_xb, estimator_ann]
#hyperpar_selection_list = [hyperpar_rf, hyperpar_xb, None]

m = LandMapper(points=data, 
    feat_cols = covs_rfe, 
    calibration_idx = calibration_idx,
    target_col = target_column, 
    estimator_list = estimator_list, 
    hyperpar_selection_list = hyperpar_selection_list,
    meta_estimator = meta_estimator,
    cv = cv_method,
    cv_njobs=int(cv_njobs),
    pred_method='predict_proba',
    weight_col=weight_column,
    #feature_weights=feature_weights,
    cv_group_col = spatial_cv_column,
    n_jobs = 10,
    verbose = True)

m.train()

print(metrics.classification_report(
    m.target,
    np.argmax(m.eval_pred, axis=1)
))

fn_landmapper =  'model_v20240210/landmapper_100.lz4' 
m.save_instance(fn_landmapper)

ttprint("End")