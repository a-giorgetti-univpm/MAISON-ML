import os
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.svm import SVR, SVC
from lightgbm import LGBMClassifier
from scikeras.wrappers import KerasRegressor   ### KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_sequences_by_participant(df, window_size, feature_cols, target_col):
    X_list, y_list, groups_list = [], [], []
    
    for p_id, group in df.groupby("participant"):
        # Se il paziente ha abbastanza dati per almeno una finestra
        if len(group) >= window_size:
            features = group[feature_cols].values
            target = group[target_col].values
            
            # Crea sequenze solo per questo paziente  (es. se window_size=5, e il paziente ha 10 campioni, creerà 6 sequenze)
            for i in range(len(group) - window_size):
                X_list.append(features[i : i + window_size])  ## con i = 0 e window_size = 5, ottieni gli indici 0, 1, 2, 3, 4, per un totale di 5 elementi. L’indice 5 è escluso.
                y_list.append(target[i + window_size])
                groups_list.append(p_id) # Salva il partecipante corretto
                
    return np.array(X_list), np.array(y_list), np.array(groups_list)


def build_lstm_model(n_features, timesteps, n_outputs=3, n_units=16, dropout=0.2, lr=1e-3):
    """Regressione multi-uscita: SIS, OHS, OKS (n_outputs=3)."""
    model = Sequential([
        LSTM(n_units, input_shape=(timesteps, n_features)),  ## n_units: 32, 64
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(n_outputs),
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse",        ### funzione obiettivo da minimizzare
        metrics=["mae"],   ### serve per monitorare le prestazioni del modello durante l'addestramento e la valutazione
    )

    return model


# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

# Define root directory
root = '.'

df = pd.read_csv('./new_dataset/maison-llf-features.csv', sep=",")

ana = pd.read_csv('./new_dataset/maison-llf-demographics.csv', sep=",")

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
    "motion-max-timestamp",
    "step-max-timestamp",
    "SISS_Category_Q",
    "OHSS_Category_Q",
    "OKSS_Category_Q"
    #"sis",
    #"ohs",
    #"oks"
]

feature_cols = [c for c in data.columns if c not in exclude_cols]
X = data[feature_cols].select_dtypes(include=[np.number]).copy()
groups = data["participant"]

# Conta i record per ogni partecipante
counts = df.groupby("participant").size()

WINDOW_SIZE = 56

nct = df.groupby('participant')['clinical-timestamp'].nunique()

# Define classifier and hyperparameter grid
param_grid = {
            "model__n_units": [16, 32, 64],
            "model__dropout": [0.0, 0.2],
            "model__lr": [1e-3, 3e-4],
            "batch_size": [16, 32],
            "epochs": [10, 20],
        }

# Leave-One-Patient-Out CV (LOPO)
outer_logo = LeaveOneGroupOut()

# Metriche per fold (regressione su SIS, OHS, OKS)
SCORE_TARGETS = ["SIS", "OHS", "OKS"]
performance_metrics = []

# ESEMPIO D'USO:
WINDOW_SIZE = 7 # 4-14-28-56

lista_features = X.columns

X_input, y_input, groups_input = create_sequences_by_participant(
    data, WINDOW_SIZE, lista_features, ['sis', 'ohs', 'oks']
)

# Ora le lunghezze saranno TUTTE uguali e coerenti
# print(X_input.shape[0], y_input.shape[0], groups_input.shape[0])
### 49 sequenze per paziente * 18 pazienti = 882 (di 7 campioni ciascuno, cioè la window size; 90 sono i predittori)

# Outer LOPO Loop
count=0
y_input = y_input 

for train_idx, test_idx in outer_logo.split(X_input, y_input, groups_input):
    #print(train_idx) index
    count=count+1
    #print(count)
    
    X_train_outer, X_test = X_input[train_idx], X_input[test_idx] #X_train_outer, X_test = X_input.loc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
    y_train_outer, y_test = y_input[train_idx], y_input[test_idx] #y_train_outer, y_test = y_input.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
    groups_train_outer = groups_input[train_idx] #groups_train_outer = groups.iloc[train_idx]
    #print(np.unique(groups_train_outer)) #print(groups_train_outer.unique())
    
    # Inner LOPO for Hyperparameter Optimization
    inner_logo = LeaveOneGroupOut()
    best_model = None
    best_score = -np.inf
    
    for inner_train_idx, inner_val_idx in inner_logo.split(X_train_outer, y_train_outer, groups_train_outer):
        X_train_inner, X_val = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
        y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
        groups_train_inner = groups_train_outer[inner_train_idx] #groups_train_inner = groups_train_outer.iloc[inner_train_idx]
        #print(np.unique(groups_train_inner)) #print(groups_train_inner.unique())
        
        # Hyperparameter tuning (massimizza R² medio sul validation set)
        for params in ParameterGrid(param_grid):
            params = {k: int(v) if isinstance(v, np.generic) else v for k, v in params.items()}
            model = KerasRegressor(
                model=build_lstm_model,
                n_features=len(lista_features),
                timesteps=WINDOW_SIZE,
                n_outputs=y_input.shape[1],  ### numero di variabili risposta da predire
                #epochs=10,
                #batch_size=32,
                verbose=0,
            )
            model.set_params(**params)
            model.fit(X_train_inner, y_train_inner)
            y_val_pred = model.predict(X_val)
            score = r2_score(y_val, y_val_pred, multioutput="uniform_average")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

    best_model.fit(X_train_outer, y_train_outer)
    y_pred = best_model.predict(X_test)

    mae_each = mean_absolute_error(y_test, y_pred, multioutput="raw_values")  #### [MAE_SIS, MAE_OHS, MAE_OKS]
    rmse_avg = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))  ##uniform_average
    r2_avg = r2_score(y_test, y_pred, multioutput="raw_values")   ##uniform_average

    performance_metrics.append(np.concatenate([mae_each, rmse_avg, r2_avg]))

metric_cols = [f"MAE_{n}" for n in SCORE_TARGETS] + [f"RMSE_{n}" for n in SCORE_TARGETS] + [f"R2_{n}" for n in SCORE_TARGETS]  ### ["RMSE_avg", "R2_avg"]
performance_df = pd.DataFrame(performance_metrics, columns=metric_cols)

# Salva metriche di regressione (SIS, OHS, OKS)
output_path = os.path.join(root, "new_results/LSTM_scores.xlsx")
with pd.ExcelWriter(output_path) as writer:
    performance_df.to_excel(writer, sheet_name="All_Folds")