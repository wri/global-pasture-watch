from hummingbird.ml import convert
from pathlib import Path
from skmap.mapper import LandMapper
from tl2cgen.contrib.util import _libext
import hummingbird
import numpy as np
import os
import pathlib
import tl2cgen
import treelite

fn_landmapper =  'model_v20240210/landmapper_100.lz4'
m = LandMapper.load_instance(fn_landmapper)

#years = range(2000,2022 + 1,2)
#n_pixel = 16000000 * len(years)
# i = 100
i = 10
n_pixel = 16000000

X = np.concatenate([m.features for i in range(0,i)], axis=0)[0:n_pixel,:]
X_meta = np.concatenate([m.meta_features for i in range(0,i)], axis=0)[0:n_pixel,:]

model_rf = m.estimator_list[0]
model_xgb = m.estimator_list[1]
model_ann = m.estimator_list[2]

dtrain = tl2cgen.DMatrix(X, dtype="float32")

tmpdir = Path('./model_v20240210/compiled')
tmpdir.mkdir(parents=True, exist_ok=True)

####################################################
###### RF
####################################################

model = treelite.sklearn.import_model(model_rf)
annotation_path = tmpdir.joinpath(f"{prefix}_rf_annotation.json")
tl2cgen.annotate_branch(model, dtrain, path=annotation_path, verbose=True)

libpath = pathlib.Path(tmpdir) / (f"{prefix}_rf" + _libext())
tl2cgen.export_lib(
  model,
  toolchain= '/opt/intel/oneapi/compiler/2024.0/bin/icx-cc', # Intel Compiler
  libpath=libpath,
  params={"annotate_in": annotation_path, "parallel_comp": 60, "verbose": 1, 'quantize': 1},
  verbose=True,
)

####################################################
###### XGB
####################################################

model_xgb.save_model(str(tmpdir.joinpath(f"{prefix}_xgb.bin")))

#model_xgb.objective = 'multi:softprob'
#model = treelite.Model.from_xgboost(model_xgb.get_booster())

#annotation_path = tmpdir.joinpath(f"{prefix}_xgb_annotation.json")
#tl2cgen.annotate_branch(model, dtrain, path=annotation_path, verbose=True)

#libpath = pathlib.Path(tmpdir) / (f"{prefix}_xgb" + _libext())
#tl2cgen.export_lib(
#        model,
#        toolchain= '/opt/intel/oneapi/compiler/2024.0/bin/icx-cc',  # Intel Compiler
#        libpath=libpath,
#        params={"annotate_in": annotation_path, "parallel_comp": clf.n_estimators, "verbose": 1, 'quantize': 1}, # 
#        verbose=True,
#    )

####################################################
###### ANN
####################################################

#ann_hb = convert(m.estimator_list[2], torch.jit.__name__, X)
#ann_hb = convert(m.estimator_list[2], 'torchscript', X)
#extra_config = {hummingbird.ml.operator_converters.constants.BATCH_SIZE: 16000000}
#hb_model = hummingbird.ml.convert(knn_model, "tpyorch", extra_config=extra_config)

ann_hb = convert(model_ann, 'pytorch') # Fastest loading and prediction time
ann_hb.save(str(tmpdir.joinpath(f"{prefix}_ann")))

####################################################
###### Meta-learner
####################################################

met_hb = convert(m.meta_estimator, 'pytorch')
met_hb.save(str(tmpdir.joinpath(f"{prefix}_logreg")))