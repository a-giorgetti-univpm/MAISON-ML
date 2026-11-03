from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd


ANA_QUAL_COLS = ['sex', 'fracture-type', 'relationship', 'education', 'work', 'ethnicity']



def build_preprocessor(num_cols):
    """
    Build the preprocessing block for numeric and categorical features.

    This is created inside each LOPO fold so all preprocessing is fit on
    training participants only.
    """
    # num_cols = SAFE_SENSOR_COLS + ANA_QUAN_COLS   # sensor-only

    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),  # SimpleImputer di scikit-learn riempie i valori mancanti (NaN) di una tabella con una statistica calcolata sulle colonne
        ('scale',  RobustScaler()),  # RobustScaler porta ogni colonna su una scala comune usando mediana e scarto interquartile
    ])

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OrdinalEncoder(  ## applica un OrdinalEncoder (ordine alfabetico)  ## TODO: non mi piace, provare OneHotEncoder
            handle_unknown='use_encoded_value',   ## handle_unknown='use_encoded_value' dice cosa fare se in transform compare una categoria mai vista in fit. Invece di sollevare un errore, le scrive unknown_value=-1
            unknown_value=-1, 
        )),
    ])

    return ColumnTransformer(  ## applica trasformazioni alle colonne
        transformers=[
            ('num', numeric_pipe, num_cols),
            ('cat', cat_pipe, ANA_QUAL_COLS),
        ],
        remainder='drop',   # drop metadata and non-feature columns (butta le colonne non elencate, come participant e timestamp)
    )



def build_model_pipeline(model, num_col):
    """
    Build a full pipeline: preprocessing + model.
    """
    steps = [('preprocess', build_preprocessor(num_col))]  ## steps è una lista
    steps.append(('model', model))
    return Pipeline(steps)



def _ols_slope(x):
    """
    Compute the slope against a simple time index.
    """
    n = len(x)
    if n < 2:
        return np.nan
    t   = np.arange(n, dtype=np.float64)
    t_c = t - t.mean()
    d   = np.dot(t_c, t_c)  ## somma dei quadrati dei numeri in t_c
    if d == 0.0:
        return np.nan
    return np.dot(t_c, x - x.mean()) / d   ## sarebbe la slope della regressione lineare tra t (time) e x (variabile)



def make_features(df, WINDOWS, PARTICIPANT_COL, DATE_COL, SAFE_SENSOR_COLS):
    """
    Add rolling-window features for each participant after sorting by time.
    """
    windows = WINDOWS

    # Sort by participant and time before computing rolling features
    df = df.sort_values([PARTICIPANT_COL, DATE_COL]).reset_index(drop=True)

    parts = []
    for _, grp in df.groupby(PARTICIPANT_COL, sort=False): # _ è l'id partecipante
        grp = grp.sort_values(DATE_COL).reset_index(drop=True)
        new_cols = {}
        for W in windows:
            for col in SAFE_SENSOR_COLS:
                s         = grp[col].astype(np.float64)
                roll_min1 = s.rolling(window=W, min_periods=1)  ### prepara una finestra mobile: per ogni giorno, la finestra prende quel giorno e i W - 1 precedenti. Con W = 7, il giorno 10 usa i giorni 4, 5, 6, 7, 8, 9 e 10.
                roll_min2 = s.rolling(window=W, min_periods=2)
                new_cols[f'{col}__roll{W}_mean']  = roll_min1.mean()  # se usassi solo min_periods=2, media, minimo e massimo del primo giorno diventerebbero NaN, perchè avrei 1 (dato) < 2 (min_periods)
                new_cols[f'{col}__roll{W}_std']   = roll_min2.std()
                new_cols[f'{col}__roll{W}_min']   = roll_min1.min()
                new_cols[f'{col}__roll{W}_max']   = roll_min1.max()
                new_cols[f'{col}__roll{W}_slope'] = roll_min2.apply(
                    _ols_slope, raw=True
                )
        parts.append(pd.concat([grp, pd.DataFrame(new_cols, index=grp.index)], axis=1))  ## in parts: un blocchetto per ogni partecipante
        ## grp.index è un indice nuovo, creato dentro il ciclo, non è l'id del partecipante
        ## le colonne di grp, compresa participant, (e le righe)restano com'erano, non essendoci una funzione agg che schiaccia la porzione di df (grp) su una riga (rendendo il partecipante unico/chiave)

    return pd.concat(parts, ignore_index=True)



def quartile_creation(df_model):

    # Compute quartiles for discretization
    siss_q1, siss_q2, siss_q3 = np.percentile(df_model["sis"], [25, 50, 75])
    ohss_q1, ohss_q2, ohss_q3 = np.percentile(df_model["ohs"], [25, 50, 75])
    okss_q1, okss_q2, okss_q3 = np.percentile(df_model["oks"], [25, 50, 75])

    # Define quartile-based bins and labels
    quartile_labels = [0, 1, 2, 3]

    # Apply discretization
    df_model["SISS_Category_Q"] = pd.cut(df_model["sis"], bins=[df_model["sis"].min(), siss_q1, siss_q2, siss_q3, df_model["sis"].max()],
                                labels=quartile_labels, include_lowest=True).astype(int)
    df_model["OHSS_Category_Q"] = pd.cut(df_model["ohs"], bins=[df_model["ohs"].min(), ohss_q1, ohss_q2, ohss_q3, df_model["ohs"].max()],
                                labels=quartile_labels, include_lowest=True).astype(int)

    df_model["OKSS_Category_Q"] = pd.cut(df_model["oks"], bins=[df_model["oks"].min(), okss_q1, okss_q2, okss_q3, df_model["oks"].max()],
                                labels=quartile_labels, include_lowest=True).astype(int)

    return df_model



def delta_creation(df, df_model, PARTICIPANT_COL, DATE_COL, CLINICAL_TS_COL, TARGET_SIS, TARGET_OHS, TARGET_OKS):

    _raw_clin = df[[PARTICIPANT_COL, DATE_COL, CLINICAL_TS_COL]].copy()
    _raw_clin[CLINICAL_TS_COL] = pd.to_datetime(_raw_clin[CLINICAL_TS_COL])

    # Add clinical timestamp back so the earliest block can be identified
    _df_clin = (
        df_model[[PARTICIPANT_COL, DATE_COL, TARGET_SIS, TARGET_OHS, TARGET_OKS]]
        .merge(_raw_clin, on=[PARTICIPANT_COL, DATE_COL], how='left')
    )

    # Take the first clinical block for each participant
    _baselines = (
        _df_clin
        .sort_values([PARTICIPANT_COL, CLINICAL_TS_COL])
        .groupby(PARTICIPANT_COL, sort=False)[[TARGET_SIS, TARGET_OHS, TARGET_OKS]]
        .first()
        .rename(columns={TARGET_SIS: 'baseline_sis', TARGET_OHS: 'baseline_ohs',
                        TARGET_OKS: 'baseline_oks'})
        .reset_index()
    )

    print("Participant baselines (SIS / OHS / OKS at earliest clinical block):")
    print(_baselines.to_string(index=False))

    TARGET_SIS_DELTA = 'sis_delta'
    TARGET_OHS_DELTA = 'ohs_delta'
    TARGET_OKS_DELTA = 'oks_delta'

    df_model_delta = (
        df_model.copy()
        .merge(_baselines, on=PARTICIPANT_COL, how='left')
    )
    df_model_delta[TARGET_SIS_DELTA] = (
        df_model_delta[TARGET_SIS] - df_model_delta['baseline_sis']
    ).astype(float)
    df_model_delta[TARGET_OHS_DELTA] = (
        df_model_delta[TARGET_OHS] - df_model_delta['baseline_ohs']
    ).astype(float)
    df_model_delta[TARGET_OKS_DELTA] = (
        df_model_delta[TARGET_OKS] - df_model_delta['baseline_oks']
    ).astype(float)
    df_model_delta = df_model_delta.drop(columns=['baseline_sis', 'baseline_ohs', 'baseline_oks'])

    print()
    for _dtgt in [TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA]:
        _rng = df_model_delta[_dtgt]
        print(f"  {_dtgt}: min={_rng.min():.1f}  max={_rng.max():.1f}  "
            f"mean={_rng.mean():.2f}  std={_rng.std():.2f}  n_unique={_rng.nunique()}")

    print(f"\n[OK] df_model_delta ready: {df_model_delta.shape}")


    return df_model_delta, TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA
