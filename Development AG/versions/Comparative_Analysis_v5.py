import os
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import json
import torch
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
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, SVC
# CatBoost
from catboost import CatBoostClassifier
# LightGBM
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline

import numba
from sktime.transformations.panel.rocket import Rocket
from sklearn.base import clone
# alternativa più veloce spesso: MiniRocket (se disponibile nella tua versione)
# from sktime.transformations.panel.rocket import MiniRocket

import logging
import time

# Logger per timing
timing_logger = logging.getLogger("timing")
timing_logger.setLevel(logging.INFO)
timing_handler = logging.FileHandler("timing.log", mode='w')
timing_formatter = logging.Formatter("%(asctime)s - %(message)s")
timing_handler.setFormatter(timing_formatter)
timing_logger.addHandler(timing_handler)


def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def build_lstm_model(n_features, timesteps=1, n_units=64, dropout=0.2, lr=1e-3):

    model = Sequential([
        LSTM(64, input_shape=(timesteps, n_features)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    #model.summary()
    return model


def build_lstm_model_OLD(n_features, n_units=32, dropout=0.2, lr=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, n_features)),  # (timesteps, features)
        tf.keras.layers.LSTM(n_units, dropout=dropout),
        tf.keras.layers.Dense(1, activation="sigmoid")  # binaria
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_tcn_model(n_features, n_filters=32, kernel_size=3, lr=1e-3):
    model = Sequential([
    TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            input_shape=(timesteps, n_features)
        ),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
                    ])
    model.compile(
                    optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
    
    #model.summary()
    return model


def inner_LoPo_ParOpt(model, param_grid, X_train_outer, y_train_outer, groups_train_outer, mod):
    
    # Inner LOPO for Hyperparameter Optimization
    inner_logo = LeaveOneGroupOut()
    best_model = None
    best_score = -np.inf

    for params in ParameterGrid(param_grid):
       
        params = {k: int(v) if isinstance(v, np.generic) else v for k, v in params.items()}
        model_clone = clone(model)       # crea copia pulita del modello
        model_clone.set_params(**params)

        fold_scores = []

        for inner_train_idx, inner_val_idx in inner_logo.split(X_train_outer, y_train_outer, groups_train_outer):
            X_train_inner, X_val = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
            y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
            #groups_train_inner = groups_train_outer.iloc[inner_train_idx]
            #print(groups_train_inner.unique())

            # un ciclo per ogni gruppo/paziente nel valutation setting

            model_clone.fit(X_train_inner, y_train_inner)
            y_val_pred = model_clone.predict(X_val)
            score = np.nan
            
            if mod == 'classification':
                score = f1_score(y_val, y_val_pred, average="macro")
            else:
                score = r2_score(y_val, y_val_pred)

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)  ### media su tutti i pazienti che ruotano a giro nel validation per quella combinazione di parametri

        if mean_score > best_score:
            best_score = mean_score
            best_model = model_clone
            best_params = params  # Store the best hyperparameters
                
    return best_model, best_params, best_score
    

def inner_GridSearch_ParOpt(estimator, param_grid, X_train_outer, y_train_outer, groups_train_outer, mod):

    print(param_grid)
    inner_cv = LeaveOneGroupOut()
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1
    )
    
    grid.fit(X_train_outer, y_train_outer, groups=groups_train_outer)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def save_ConfMat(overall_conf_matrix, root):

    # Plot overall confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(overall_conf_matrix, annot=True, cmap="coolwarm", xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Overall Confusion Matrix OHSS")
    conf_matrix_path = os.path.join(root, "results/overall_confusion_matrix_ohss.pdf")
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    return 0

#print(f"\n✅ Performance metrics saved as: {output_path}")
#print(f"✅ Overall Confusion Matrix saved as: {conf_matrix_path}")


# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

# Define root directory
root = '.'

df = pd.read_csv('./new_dataset/maison-llf-features.CSV', sep=",")  ### maison-llf-features_TEST.CSV

ana = pd.read_csv('./new_dataset/maison-llf-demographics.CSV', sep=",")  ### maison-llf-demographics_TEST

ana_col = list(ana.columns)

ana_encoded = ana[["participant", "age", "sex", "education", "work", "fracture-type", "ethnicity"]].copy()

# male=1, female=0
ana_encoded["sex_male"] = ana_encoded["sex"].map({"male": 1, "female": 0})

# Education label encoding with doctorate as highest level
education_order = {
    "secondary education": 0,
    "undergraduate degree": 1,
    "graduate degree": 2,
    "doctorate degree": 3
}
ana_encoded["education_label"] = ana_encoded["education"].map(education_order)

# retired=0, employed part-time=1
ana_encoded["work_part_time"] = ana_encoded["work"].map({"retired": 0, "employed part-time": 1})

fracture_dummies = pd.get_dummies(
    ana_encoded["fracture-type"],
    prefix="fracture",
    dtype=int
)

ethnicity_dummies = pd.get_dummies(
    ana_encoded["ethnicity"],
    prefix="ethnicity",
    dtype=int
)

num_ana = pd.concat(
    [
        ana_encoded[["participant", "age", "sex_male", "education_label", "work_part_time"]],
        fracture_dummies,
        ethnicity_dummies,
    ],
    axis=1,
)

data = df.merge(num_ana, how='right', on='participant')

# Compute quartiles for discretization
siss_q1, siss_q2, siss_q3 = np.percentile(data["sis"], [25, 50, 75])
ohss_q1, ohss_q2, ohss_q3 = np.percentile(data["ohs"], [25, 50, 75])
okss_q1, okss_q2, okss_q3 = np.percentile(data["oks"], [25, 50, 75])

# Define quartile-based bins and labels
quartile_labels = [0, 1, 2, 3]

# Apply discretization
data["SISS_Category_Q"] = pd.cut(data["sis"], bins=[data["sis"].min(), siss_q1, siss_q2, siss_q3, data["sis"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)
data["OHSS_Category_Q"] = pd.cut(data["ohs"], bins=[data["ohs"].min(), ohss_q1, ohss_q2, ohss_q3, data["ohs"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)

data["OKSS_Category_Q"] = pd.cut(data["oks"], bins=[data["oks"].min(), okss_q1, okss_q2, okss_q3, data["oks"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)

# Extract only numeric features for LOPO (drop timestamps/string columns).
exclude_cols = [
    "participant",
    "timestamp",
    "clinical-timestamp",
    #"motion-max-timestamp",
    #"step-max-timestamp",
    "SISS_Category_Q",
    "OHSS_Category_Q",
    "OKSS_Category_Q",
]

feature_cols = [c for c in data.columns if c not in exclude_cols]
X = data[feature_cols].select_dtypes(include=[np.number]).copy()
groups = data["participant"]

# Conta i record per ogni partecipante
counts = df.groupby("participant").size()

WINDOW_SIZE = 56

# Define classifier and hyperparameter grid

TABULAR_MODELS = [
    {
        "name": "XGBoost",
        "mode": ["classification"],
        "data_kind": "tabular",
        "estimator": XGBClassifier(
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed
        ),
        "param_grid": {
            "n_estimators": [20, 50],  ## 20, 50, 100
            "max_depth": [5, 10, 20, 50],   ## 3, 6, 8
            "learning_rate": [0.3]   ## 0.001, 0.01, 0.1
            #"subsample": [0.8, 1.0],
            #"colsample_bytree": [0.8, 1.0],
            #"reg_lambda": [1.0, 5.0],
        },
    },

    {
    "name": "XGBoost",
    "mode": ["regression"],
    "data_kind": "tabular",
    "estimator": XGBRegressor(
        eval_metric="rmse",      # oppure "mae"
        tree_method="hist",
        random_state=seed
    ),
    "param_grid": {
        "n_estimators": [20, 50],
        "max_depth": [5, 10, 20, 50],
        "learning_rate": [0.001, 0.01]   ### 0.3
        },
    },
    
    {
        "name": "CatBoost",
        "mode": ["classification"],     ###  regression: dura troppo (2026-04-16 07:19:05,337 - Model=CatBoost | Resp=ohs | Time=29408.01s)
        "data_kind": "tabular",
        "estimator": CatBoostClassifier(
            verbose=False,
            random_seed=seed
        ),
        "param_grid": {
            "iterations": [50, 100],
            "depth": [3, 6], ## 8
            "learning_rate": [0.001, 0.3]   ### 0.01, 0.1
            #"l2_leaf_reg": [3, 10],
        },
    },
    
    {
        "name": "LightGBM",
        "mode": ["classification", "regression"],
        "data_kind": "tabular",
        "estimator": LGBMClassifier(
            random_state=seed,
            verbosity=-1,
            force_col_wise=True
        ),
        "param_grid": {
            "n_estimators": [20, 50],   ### 80
            #"num_leaves": [15, 31, 63],
            "max_depth": [5, 10, 20, 50],   ## 8
            "learning_rate": [0.001, 0.01]  ### 0.01, 0.1, 0.3
            #"subsample": [0.8, 1.0],
            #"colsample_bytree": [0.8, 1.0],
        },
    },
    
    {
        "name": "DecisionTree (DT)",
        "mode": ["classification", "regression"],
        "data_kind": "tabular",
        "estimator": DecisionTreeClassifier(random_state=seed),
        "param_grid": {
            "max_depth": [5, 10, 20, 50]  ### 8
            #"min_samples_split": [2, 5, 10],
            #"min_samples_leaf": [1, 2, 5],
            #"criterion": ["gini", "entropy"],
        },
    },

    # SVR è un regressore, Support Vector Regressor (attenzione: non va con scoring f1_macro!)
    {
        "name": "SVR",
        "mode": ["regression"],
        "data_kind": "tabular",
        "estimator": SVR(),
        "param_grid": {
            "kernel": ["rbf"],  ## "poly" --> sconsigliato per fare SVR
            "C": [1, 10],
            "epsilon": [0.01, 0.2],  ### 0.1
            "gamma": ["scale"],
        },
    },

    # SVC, è un classificatore basato su SVM

    {
        "name": "SVC",
        "mode": ["classification"],
        "data_kind": "tabular",
        "estimator": SVC(probability=True),
        "param_grid": {
            "kernel": ["rbf"],
            "C": [1, 10],
            "gamma": ["scale"],
            "degree": [2, 3],
        },
    }
]

n_features = X.shape[1]

ANN_MODELS = [
    {
        "name": "LSTM",
        "mode": ["classification", "regression"],
        "data_kind": "time_series",
        "estimator": KerasClassifier(
            model=build_lstm_model,
            n_features=n_features,
            epochs=10,
            batch_size=32,
            verbose=0
        ),
        "param_grid": {
            "model__n_units": [16, 32, 64],
            "model__dropout": [0.0, 0.2],
            "model__lr": [1e-3, 3e-4],
            "batch_size": [16, 32],
            "epochs": [10, 20],
        },
    },
    {
        "name": "TCN",                                 ##Temporal Convolutional Network
        "mode": ["classification", "regression"],
        "data_kind": "time_series",
        "estimator": KerasClassifier(
            model=build_tcn_model,
            n_features=n_features,
            epochs=10,
            batch_size=32,
            verbose=0
        ),
        "param_grid": {
            "model__n_filters": [16, 32, 64],
            "model__kernel_size": [3, 5],
            "model__lr": [1e-3, 3e-4],
            "batch_size": [16, 32],
            "epochs": [10, 20],
        },
    },
]

OTHER_MODELS = [
    {
        "name": "ROCKET + RidgeClassifier",
        "mode": ["classification", "regression"],
        "estimator": Pipeline(steps=[
            ("rocket", Rocket(random_state=seed)),
            ("clf", RidgeClassifier())
        ]),
        "param_grid": {
            "rocket__num_kernels": [2000, 10000],
            "clf__alpha": [0.1, 1.0, 10.0],
        },
    }
]

MODELS = TABULAR_MODELS #+ ANN_MODELS + OTHER_MODELS

#class_responses = ["OHSS_Category_Q", "SISS_Category_Q", "OKSS_Category_Q"]
reg_responses = ["sis", "oks"]   ### "ohs" temporaly escluded

target = {#"classification": class_responses,
          "regression": reg_responses}

# Leave-One-Patient-Out CV (LOPO): OUTER LOPO, perchè ha più senso fare la predict su un solo paziente nel testing_set avendo utilizzato una LOPO inner
outer_logo = LeaveOneGroupOut()

# Initialize results storage
#overall_conf_matrix = np.zeros((4, 4))  # Assuming 4 categories (0, 1, 2, 3)

inner_ParOpt = {"LOPO": inner_LoPo_ParOpt, 
                #"K-Fold": inner_GridSearch_ParOpt
               }

modes = ["regression"]    ### "classification"

data_types =["time_series", "tabular"]

# Outer LOPO Loop

for mod in modes:

    print(mod)

    responses = target[mod]

    output_path = os.path.join(root, "new_results/results_" + mod + ".xlsx")
    with pd.ExcelWriter(output_path) as writer:

        for resp in responses:

            y = data[resp]
            print(resp)

            performance_metrics_cla = []
            performance_metrics_reg = []

            #count=0

            for train_idx, test_idx in outer_logo.split(X, y, groups):
                    #print(train_idx)
                    #count=count+1
                    #print(count)
                    # ogni ciclo tengo fuori un paziente, su cui vado a fare la validation del best
            
                    X_train_outer, X_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
                    y_train_outer, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
                    groups_train_outer = groups.iloc[train_idx]
                    groups_test_outer = groups.iloc[test_idx]
                    patient = groups_test_outer.unique()[0]
                    print(patient)
                    print(groups_train_outer.unique())
                    
                    for model_dict in TABULAR_MODELS:
            
                        model = model_dict['estimator']
                        param_grid = model_dict['param_grid']
                        model_mod = model_dict['mode']
                        #print(param_grid)
            
                        if not mod in model_mod:
                            continue

                        print(model_dict['name'])
        
                        #if model_dict['name'] in ["CatBoost", "LightGBM", "DecisionTree (DT)"]:  ## sono tutti i modelli che mi danno problemi
                        #    continue
        
                        for param_opt in inner_ParOpt.keys():
        
                            func = inner_ParOpt[param_opt]
                            print(param_opt)
        
                            ## INNER OPTIMIZATION (best FOR each group-iteration)
                            try:
                                
                                start_time = time.time()

                                best_model, best_params, best_score = func(
                                                                            model,
                                                                            param_grid,
                                                                            X_train_outer,
                                                                            y_train_outer,
                                                                            groups_train_outer,
                                                                            mod
                                                                        )
            
                                best_model.fit(X_train_outer, y_train_outer)
                                
                                y_pred = best_model.predict(X_test)
                    
                                #if mod == "classification":
            
                                    #print(mod)
                                    # Compute metrics CLASSIFICATION
                                #    try:
                                #        f1 = f1_score(y_test, y_pred, average="macro")
                                #    except:
                                #        f1 = np.nan
                                #    balanced_acc = balanced_accuracy_score(y_test, y_pred)
                                #    recall = recall_score(y_test, y_pred, average="macro")
                                #    precision = precision_score(y_test, y_pred, average="macro")
                                #    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])          
                                
                                    # Aggregate confusion matrices
                                #    #overall_conf_matrix += conf_matrix
                                
                                #    # Store results
                                #    performance_metrics_cla.append([model_dict['name'], param_opt, f1, balanced_acc, recall, precision, json.dumps(best_params, sort_keys=True)])
            
                                if mod == "regression":
            
                                    #print(mod)
                                    # Compute metrics REGRESSION
                                    mae = mean_absolute_error(y_test, y_pred)
                                    mse = mean_squared_error(y_test, y_pred)
                                    rmse = np.sqrt(mse)
                                    r2 = r2_score(y_test, y_pred)
                                    mape = mean_absolute_percentage_error(y_test, y_pred)
                                    medae = median_absolute_error(y_test, y_pred)
                                    evs = explained_variance_score(y_test, y_pred)
                
                                    # Store results
                                    performance_metrics_reg.append([patient, model_dict['name'], param_opt, mae, mse, rmse, r2, mape, medae, evs, json.dumps(best_params, sort_keys=True)])

                                end_time = time.time()
                                duration = end_time - start_time
                                timing_logger.info(
                                                    f"Model={model_dict['name']} | "
                                                    f"Resp={resp} | "
                                                    f"Time={duration:.2f}s"
                                                    )

                            except Exception as e:
                                print(f"Errore: {e}")
                                continue
            
            #if mod == "classification":
            #    #print('group_by')
            #    performance_df_raw = pd.DataFrame(performance_metrics_cla, columns=["Model", "CV", "Macro-F1", "Balanced Accuracy", "Macro Recall", "Macro Precision", "Parameters"])
            #    performance_df = (performance_df_raw.groupby(["Model", "CV", "Parameters"])[["Macro-F1", "Balanced Accuracy", "Macro Recall", "Macro Precision"]]
            #                                                    .mean()
            #                                                    .reset_index()
            #                                                    #.rename(columns={"score": "mean_score"})
            #                                                    )

            if mod == "regression":
                #print('group_by')
                
                performance_df_raw = pd.DataFrame(performance_metrics_reg, columns=["Patient", "Model", "CV", "MAE", "MSE", "RMSE", "R2", "MAPE", "MEDAE", "EVS", "Parameters"])
                
                
                
                #performance_df = (performance_df_raw.groupby(["Model", "CV", "Parameters"])[["MAE", "MSE", "RMSE", "R2", "MAPE", "MEDAE", "EVS"]]
                #                                                .mean()
                #                                                 .reset_index()
                #                                                #.rename(columns={"score": "mean_score"})
                #                                                )

            performance_df_raw.to_excel(writer, sheet_name=resp, index=False)

            ## nel dataset finale posso avere lo stesso modello con diversi parametri che risultano ottimi su fold/pazienti diversi
            # non ho un modello per paziente perchè: groupby(["Model", "CV", "Parameters"])





