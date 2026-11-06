import DataQuality as dq
import PreProcessing as pp
import Modelling as md
import Saving as sv
import Plotting as pl

import os
from pathlib import Path

import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid, KFold, GroupKFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)
import catboost as cb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

import shap
import numba
from sktime.transformations.panel.rocket import Rocket
from sklearn.base import clone
from collections import Counter

import logging
import time

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)

seed = config["seed"]
ROOT = BASE_DIR
DATA_DIR = ROOT / config["data_dir"]
FEATURES_PATH = DATA_DIR / config["features_file"]
DEMOGRAPHICS_PATH = DATA_DIR / config["demographics_file"]
TIMING_LOG_PATH = ROOT / config["timing_log"]
OUTPUTS_DIR = ROOT / config["outputs_dir"]

# Logger per timing
timing_logger = logging.getLogger("timing")
timing_logger.setLevel(logging.INFO)
timing_handler = logging.FileHandler(TIMING_LOG_PATH, mode='w')
timing_formatter = logging.Formatter("%(asctime)s - %(message)s")
timing_handler.setFormatter(timing_formatter)
timing_logger.addHandler(timing_handler)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
#torch.manual_seed(seed)
np.random.seed(seed)

# Define directories
TABLES_DIR = OUTPUTS_DIR / 'tables'
FIGURES_DIR = OUTPUTS_DIR / 'figures'

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'maison-llf-features.csv', sep=",")  ### maison-llf-features_TEST.CSV
raw_ana = pd.read_csv(DATA_DIR / 'maison-llf-demographics.csv', sep=",")  ### maison-llf-demographics_TEST

# Convert timestamp to datetime for sorting and rolling features later
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== Files loaded ===")
print(f"  features.csv       : {df.shape[0]} rows x {df.shape[1]} cols")
print(f"  demographics.csv   : {raw_ana.shape[0]} rows x {raw_ana.shape[1]} cols")
print()

num_ana = raw_ana # pp.ana_encoding(raw_ana)  ### TODO

# Merge demographics into the feature table
df_merge = df.merge(num_ana, how='right', on='participant')

rc_merge = dq.merge_check(df_merge, df)

print("=" * 60)
print("BASIC INPUT DATA CHECKS")
print("=" * 60)

rc_quality = dq.initial_check(df, num_ana, df_merge)

# Set column names used throughout the notebook

PARTICIPANT_COL = 'participant'
DATE_COL        = 'timestamp'
CLINICAL_TS_COL = 'clinical-timestamp'
TARGET_SIS      = 'sis'
TARGET_OHS      = 'ohs'
TARGET_OKS      = 'oks'    # Oxford Knee Score total
WINDOWS         = [3, 7, 14]
# Questionnaire sub-items that directly sum to the targets
SIS_SUBITEMS = [f'sis-0{i}' for i in range(1, 7)]
OHS_SUBITEMS = [f'ohs-{str(i).zfill(2)}' for i in range(1, 13)]     # ohs-01..ohs-12
### zfill(2) riempie la stringa a sinistra con zeri finché non arriva a 2 caratteri
OKS_SUBITEMS   = [f'oks-{str(i).zfill(2)}' for i in range(1, 13)]   # oks-01..oks-12
CLINICAL_TESTS = ['tug', 'chairstand']
# Columns always removed before modelling
COLS_DROP_ALWAYS = (
    SIS_SUBITEMS
    + OHS_SUBITEMS
    + OKS_SUBITEMS
    + CLINICAL_TESTS
    + [CLINICAL_TS_COL]
)
# Sensor columns grouped by modality
SENSOR_MODALITIES = {
    'acceleration': [c for c in df.columns if c.startswith('acceleration')],
    'heartrate'   : [c for c in df.columns if c.startswith('heartrate')],
    'motion'      : [c for c in df.columns if c.startswith('motion')],
    'position'    : [c for c in df.columns if c.startswith('position')],
    'sleep'       : [c for c in df.columns if c.startswith('sleep')],
    'step'        : [c for c in df.columns if c.startswith('step')],
}
SAFE_SENSOR_COLS = [c for cols in SENSOR_MODALITIES.values() for c in cols]
# Demographic columns used later in the preprocessing pipeline
ANA_QUAL_COLS = ['sex', 'fracture-type', 'relationship', 'education', 'work', 'ethnicity']
ANA_QUAN_COLS = ['age']
DEMO_FEAT_COLS = ANA_QUAN_COLS + ANA_QUAL_COLS   # ['age'] + 6 categoricals = 7 features

rc_log = dq.col_log(COLS_DROP_ALWAYS, SAFE_SENSOR_COLS, SENSOR_MODALITIES, ANA_QUAL_COLS, ANA_QUAN_COLS)

# Drop leakage columns
cols_present_to_drop = [c for c in COLS_DROP_ALWAYS if c in df_merge.columns]
cols_not_found       = [c for c in COLS_DROP_ALWAYS if c not in df_merge.columns]

data = df_merge.drop(columns=COLS_DROP_ALWAYS, errors='ignore').copy()
#print("Remaining columns:")
#print(list(data.columns))

df_invest = data.copy()
df_invest[CLINICAL_TS_COL] = df_merge[CLINICAL_TS_COL]  ## rimosso sopra

df_model = dq.manage_target(df_invest, data, df_merge, PARTICIPANT_COL, CLINICAL_TS_COL, TARGET_SIS, TARGET_OHS, TARGET_OKS)

rc_report = dq.final_report(df_model, SAFE_SENSOR_COLS, TARGET_SIS, TARGET_OHS, TARGET_OKS, 
                       SIS_SUBITEMS, OHS_SUBITEMS, OKS_SUBITEMS, CLINICAL_TESTS, CLINICAL_TS_COL,
                       PARTICIPANT_COL, DATE_COL, SENSOR_MODALITIES)

FEAT_COLS = SAFE_SENSOR_COLS + ANA_QUAN_COLS + ANA_QUAL_COLS
# Rolling feature names
ROLLED_SENSOR_COLS = [
    f'{col}__roll{W}_{stat}'
    for col in SAFE_SENSOR_COLS
    for W    in WINDOWS
    for stat in ['mean', 'std', 'min', 'max', 'slope']
]
# Feature lists used later in the LOPO pipeline
ALL_NUM_COLS     = SAFE_SENSOR_COLS + ROLLED_SENSOR_COLS + ANA_QUAN_COLS
FEAT_COLS_ROLLED = ALL_NUM_COLS + ANA_QUAL_COLS

## df_modelq = pp.quartile_creation(df_model)  ## TODO: creazione dei quartili

# Hyperparameter grids used in the inner CV search
with open("param_v2.json", "r", encoding="utf-8") as file:
    parameters = json.load(file)

# Main LOPO loop:
# - hold out one participant
# - merge demographics inside the fold
# - build rolling features for train and test separately
# - run baselines and tuned models
# - store fold results

logo     = LeaveOneGroupOut()
groups   = df_model[PARTICIPANT_COL].values
all_pids = sorted(df_model[PARTICIPANT_COL].unique())
n_folds  = len(all_pids)   ### fold = participant
MAX_FOLDS = n_folds
print("=== LOPO-CV Outer Loop (Steps 4-7) ===")
print(f"  Total participants  : {n_folds}")
print(f"  Folds to run        : {MAX_FOLDS}")
print(f"  Rolling windows     : {WINDOWS} days")

rc_hy_grid = md.print_hyperparameter_grids(parameters)

num_ana.drop(columns=['age'], inplace=True)

fold_records, oof_pred_store = md.run_models(num_ana, df_model, logo, groups, n_folds, MAX_FOLDS, PARTICIPANT_COL, 
                                             WINDOWS, TARGET_SIS, TARGET_OHS, TARGET_OKS, ANA_QUAN_COLS, ANA_QUAL_COLS, 
                                             DEMO_FEAT_COLS, ALL_NUM_COLS, FEAT_COLS_ROLLED, DATE_COL, SAFE_SENSOR_COLS, seed, parameters)

df_fold_summary = md.print_summary(fold_records, MAX_FOLDS)

rc_save_baseline = sv.save_baseline(fold_records, TARGET_SIS, TARGET_OHS, TARGET_OKS, TABLES_DIR)
df_all_metrics, summary = sv.save_long_table(fold_records, TARGET_SIS, TARGET_OHS, TARGET_OKS, TABLES_DIR)

rc_figures = pl.make_model_comparison_figures(df_all_metrics, summary, TARGET_SIS, TARGET_OHS, TARGET_OKS, FIGURES_DIR)

rc_hyp_freq = md.hyperparameter_frequency(df_fold_summary,TARGET_SIS, TARGET_OHS, TARGET_OKS)

rc_scatterplot = pl.plot_actual_predicted(oof_pred_store, TARGET_SIS, TARGET_OHS, TARGET_OKS, FIGURES_DIR)

df_model_delta, TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA = pp.delta_creation(df, df_model, PARTICIPANT_COL, 
                                                                                         DATE_COL, CLINICAL_TS_COL, 
                                                                                         TARGET_SIS, TARGET_OHS, TARGET_OKS)

logo_d    = LeaveOneGroupOut()
groups_d  = df_model_delta[PARTICIPANT_COL].values
n_folds_d = len(df_model_delta[PARTICIPANT_COL].unique())
MAX_FOLDS_D = n_folds_d
print("=== Delta-Target LOPO-CV ===")
print(f"  Folds    : {MAX_FOLDS_D}")
print(f"  Targets  : {TARGET_SIS_DELTA}, {TARGET_OHS_DELTA}, {TARGET_OKS_DELTA}")

delta_records = md.run_delta_model(df_model_delta, num_ana, parameters, logo_d, groups_d, MAX_FOLDS_D, WINDOWS, ANA_QUAL_COLS, 
                                   PARTICIPANT_COL, ALL_NUM_COLS, FEAT_COLS_ROLLED, TARGET_SIS_DELTA, TARGET_OHS_DELTA, 
                                   TARGET_OKS_DELTA, DATE_COL, SAFE_SENSOR_COLS, seed)

df_delta_metrics, delta_summary, delta_per_path, delta_sum_path =sv.save_delta_summary(delta_records, TARGET_SIS_DELTA, 
                                                                                       TARGET_OHS_DELTA, TARGET_OKS_DELTA, TABLES_DIR)

fc_delta_figures = pl.make_delta_comparison_figures(delta_summary, df_delta_metrics, 
                                                    TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA, FIGURES_DIR)

rc_saved_outputs = sv.check_delta_outputs_saved(delta_per_path, delta_sum_path, FIGURES_DIR)