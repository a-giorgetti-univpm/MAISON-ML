import PreProcessing as pp

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


import pandas as pd
import numpy as np
from collections import Counter




def print_hyperparameter_grids(param_grid):
    print("=== Hyperparameter Grids ===")

    for model_name, param_grid in param_grid.items():
        print(f"\n{model_name}:")
        for param_name, param_list in param_grid.items():
            print(f"  {param_name}: {param_list}")
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)    ### n_combos *= len(v) significa n_combos = n_combos * len(v)
        print(f"  {model_name:<6}: {n_combos} combination(s)")

    return 0



def sanity_check(train_pids,test_pids, train_idx,test_idx, df_model, fold_idx, held_out_pid, n_folds, df_train, df_test, MAX_FOLDS):
    """
    Perform basic sanity checks for the fold.
    """
    ## &: intersezione tra i due set; fold_idx+1: enumerate parte da 0 nel numerare le Fold
    assert len(train_pids & test_pids) == 0, \
        f"Fold {fold_idx+1}: participant overlap"
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, \
        f"Fold {fold_idx+1}: index overlap"
    assert len(test_pids) == 1, \
        f"Fold {fold_idx+1}: expected 1 test pid"
    assert len(train_pids) == n_folds - 1, \
        f"Fold {fold_idx+1}: expected {n_folds-1} train pids"
    assert len(train_idx) + len(test_idx) == len(df_model), \
        f"Fold {fold_idx+1}: data count mismatch"

    print(f"  Fold {fold_idx+1:02d}/{MAX_FOLDS}"
        f" | held-out P{held_out_pid:02d}"
        f" | train: {len(train_pids)} pids / {len(df_train)} rows"
        f" | test: {len(df_test)} rows")

    return 0



def untuned_ridge(df_tr_fe, df_te_fe, fold_rec, TARGET_SIS, TARGET_OHS, TARGET_OKS, ALL_NUM_COLS, FEAT_COLS_ROLLED, ANA_QUAL_COLS, seed):
    # Untuned Ridge on all rolled features
    for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
        pipe = pp.build_model_pipeline(
            Ridge(alpha=1.0, random_state=seed),
            ALL_NUM_COLS                  ## ALL_NUM_COLS = SAFE_SENSOR_COLS + ROLLED_SENSOR_COLS + ANA_QUAN_COLS
                                          ## ANA_QUAL_COLS   
        )
        pipe.fit(df_tr_fe[FEAT_COLS_ROLLED], df_tr_fe[target])  ## la trasformazione riguarda solo il primo input perchè per la y (df_tr_fe[target]) il ColumnTransformer non trova i campi da trasformare
    
        y_pred = pipe.predict(df_te_fe[FEAT_COLS_ROLLED])       ## FEAT_COLS_ROLLED = ALL_NUM_COLS + ANA_QUAL_COLS

        mae = mean_absolute_error(df_te_fe[target], y_pred)
        r2  = r2_score(df_te_fe[target], y_pred)
        fold_rec[f'ridge_mae_{target}'] = round(mae, 4)
        fold_rec[f'ridge_r2_{target}']  = round(r2,  4)
        print(f"    -> [{target.upper()}] Ridge(sensor+demo)  MAE={mae:.3f}  R2={r2:.3f}")

    return 0



def baseline_a(df_tr, df_te, fold_rec, TARGET_SIS, TARGET_OHS, TARGET_OKS, ANA_QUAN_COLS, DEMO_FEAT_COLS, ANA_QUAL_COLS, seed):
    # Baseline A: demographics-only Ridge
    for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
        pipe_a = pp.build_model_pipeline(
            Ridge(alpha=1.0, random_state=seed),
            ANA_QUAN_COLS
            ## ANA_QUAL_COLS         # scale only 'age'
        )
        pipe_a.fit(df_tr[DEMO_FEAT_COLS], df_tr[target])  ## DEMO_FEAT_COLS = ANA_QUAN_COLS + ANA_QUAL_COLS   # ['age'] + 6 categoricals = 7 features

        y_pred_a = pipe_a.predict(df_te[DEMO_FEAT_COLS])

        mae_a = mean_absolute_error(df_te[target], y_pred_a)
        r2_a  = r2_score(df_te[target], y_pred_a)
        fold_rec[f'bl_a_mae_{target}'] = round(mae_a, 4)
        fold_rec[f'bl_a_r2_{target}']  = round(r2_a,  4)
        print(f"    -> [{target.upper()}] Baseline-A(demo)    MAE={mae_a:.3f}  R2={r2_a:.3f}")

    return 0



def _merge_demographics(df, demographics, participant_col):
    """Add missing demographics without suffixing columns already in the fold."""
    missing_cols = [
        col for col in demographics.columns
        if col != participant_col and col not in df.columns
    ]
    return df.merge(
        demographics[[participant_col] + missing_cols],
        on=participant_col,
        how='left',
        validate='many_to_one',
    )



def run_models(num_ana, df_model, logo, groups, n_folds, MAX_FOLDS, PARTICIPANT_COL, WINDOWS, TARGET_SIS, TARGET_OHS, TARGET_OKS, ANA_QUAN_COLS, ANA_QUAL_COLS, DEMO_FEAT_COLS, ALL_NUM_COLS, FEAT_COLS_ROLLED, DATE_COL, SAFE_SENSOR_COLS, seed, param_grid):

    fold_records = []

    oof_pred_store = {}  # OOF predictions: key=(model_target) -> {y_true, y_pred}

    for fold_idx, (train_idx, test_idx) in enumerate(
            logo.split(df_model, groups=groups)):  ## mette nel test tutte le righe di un paziente e nel train le righe di tutti gli altri
                                                ## LeaveOneGroupOut ordina gli id dei pazienti
        if fold_idx >= MAX_FOLDS:  ### se vuoi fermarti prima invece di fare la LOPO su ogni paziente
            break

        #if fold_idx >= 1:  ### TODO: TEST only
        #            break

        # Split into train and test
        df_train = df_model.iloc[train_idx].copy()
        df_test  = df_model.iloc[test_idx].copy()

        train_pids   = set(df_train[PARTICIPANT_COL].unique())
        test_pids    = set(df_test[PARTICIPANT_COL].unique())
        held_out_pid = int(list(test_pids)[0])  ## tira fuori il numero da test_pids

        rc_sanity_check = sanity_check(train_pids,test_pids, train_idx,test_idx, df_model, fold_idx, held_out_pid, n_folds, df_train, df_test, MAX_FOLDS)

        # Merge demographics inside the fold
        df_tr = _merge_demographics(df_train, num_ana, PARTICIPANT_COL)
        df_te = _merge_demographics(df_test, num_ana, PARTICIPANT_COL)

        # Build rolling features separately for train and test
        df_tr_fe = pp.make_features(df_tr, WINDOWS, PARTICIPANT_COL, DATE_COL, SAFE_SENSOR_COLS)
        df_te_fe = pp.make_features(df_te, WINDOWS, PARTICIPANT_COL, DATE_COL, SAFE_SENSOR_COLS)

        fold_rec = {
            'fold'         : fold_idx + 1,
            'held_out_pid' : held_out_pid,
            'n_train_pids' : len(train_pids),
            'n_train_rows' : len(df_train),
            'n_test_rows'  : len(df_test),
            #'n_features'   : len(FEAT_COLS_ROLLED),
            'assertions'   : 'PASS',
        }
        ### TODO: vedere come gestire il OneHot Encoding (altra Pipeline?)
        # Untuned Ridge on all rolled features                     
        rc_unt_ridge = untuned_ridge(df_tr_fe, df_te_fe, fold_rec, TARGET_SIS, TARGET_OHS, TARGET_OKS, 
                                     ALL_NUM_COLS, FEAT_COLS_ROLLED, ANA_QUAL_COLS, seed)

        # Baseline A: demographics-only Ridge
        rc_base_a = baseline_a(df_tr, df_te, fold_rec, TARGET_SIS, TARGET_OHS, TARGET_OKS, ANA_QUAN_COLS, DEMO_FEAT_COLS, 
                               ANA_QUAL_COLS, seed)

        # Baseline B: predict the training mean for every test row
        for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
            train_mean = float(df_tr[target].mean())
            y_te_vals  = df_te[target].values
            y_pred_b   = np.full(len(y_te_vals), train_mean)  ## costruisce un array lungo quanto il test, con ogni casella uguale a train_mean

            mae_b = mean_absolute_error(y_te_vals, y_pred_b)
            r2_b  = r2_score(y_te_vals, y_pred_b)
            fold_rec[f'bl_b_mae_{target}'] = round(mae_b, 4)
            fold_rec[f'bl_b_r2_{target}']  = round(r2_b,  4)
            print(f"    -> [{target.upper()}] Baseline-B(mean)    MAE={mae_b:.3f}  R2={r2_b:.3f}")

        # Inner CV tuning for Ridge, ElasticNet, RF, and LightGBM
        n_inner   = min(len(train_pids), 5)      ### len(train_pids) = 17
        inner_cv  = GroupKFold(n_splits=n_inner)
        inner_grp = df_tr_fe[PARTICIPANT_COL].values

        Model_SPECS = [   ## TODO: aggiungere i miei modelli in automatico
            # (tag, model_class, model_kwargs, param_grid, selector)
            ('tuned_ridge', Ridge, dict(), param_grid["PARAM_GRID_RIDGE"], None),
            ('tuned_elasticnet', ElasticNet, dict(max_iter=10000, random_state=seed), param_grid["PARAM_GRID_ELASTICNET"], None),
            ('rf', RandomForestRegressor, dict(random_state=seed, n_jobs=1), param_grid["PARAM_GRID_RF"], None),
            ('lgbm', LGBMRegressor, dict(random_state=seed, verbosity=-1, n_jobs=1), param_grid["PARAM_GRID_LGB"], None),
            ('xgb', XGBRegressor, dict(eval_metric="rmse", tree_method="hist", random_state=seed), param_grid["PARAM_GRID_XGBOOST"], None),
            ('catboost', CatBoostRegressor, dict(eval_metric="RMSE", verbose=False, random_seed=seed), param_grid["PARAM_GRID_CATBOOST"], None),
            ('dt', DecisionTreeRegressor, dict(random_state=seed), param_grid["PARAM_GRID_DECISIONTREE"], None),
            ('svr', SVR, dict(), param_grid["PARAM_GRID_SVR"], None)
        ]

        for model_tag, model_cls, model_kwargs, model_grid, selector in Model_SPECS:
            for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
                estimator = model_cls(**model_kwargs)    ## **model_kwargs apre il dizionario e passa ogni voce come argomento
                pipe = pp.build_model_pipeline(estimator, ALL_NUM_COLS)   ## ANA_QUAL_COLS
                search = GridSearchCV(
                    pipe,
                    model_grid,
                    cv=inner_cv,
                    scoring='neg_mean_absolute_error',
                    refit=True,   ### Finita la ricerca, refit=True riaddestra la pipeline vincitrice su tutti i pazienti di training di quella fold
                    n_jobs=-1,
                )
                search.fit(
                    df_tr_fe[FEAT_COLS_ROLLED],
                    df_tr_fe[target],
                    groups=inner_grp,
                )
                y_pred = search.predict(df_te_fe[FEAT_COLS_ROLLED])

                mae = mean_absolute_error(df_te_fe[target], y_pred)
                r2  = r2_score(df_te_fe[target], y_pred)
                fold_rec[f'{model_tag}_mae_{target}']  = round(mae, 4)
                fold_rec[f'{model_tag}_r2_{target}']   = round(r2,  4)
                fold_rec[f'{model_tag}_best_{target}'] = str(search.best_params_)  ## search.best_params_ è la combinazione scelta per quella fold (escludendo quel paziente)
                # Save OOF predictions for later plots
                _key = f'{model_tag}_{target}'
                if _key not in oof_pred_store:
                    oof_pred_store[_key] = {'y_true': [], 'y_pred': []}
                oof_pred_store[_key]['y_true'].extend(df_te_fe[target].tolist())  ## extend aggiunge gli elementi di una lista in coda
                oof_pred_store[_key]['y_pred'].extend(y_pred.tolist())
                print(f"    -> [{target.upper()}] {model_tag:<12}  "
                    f"MAE={mae:.3f}  R2={r2:.3f}  best={search.best_params_}")

        # Make sure the OOF keys exist for later scatter plots
        for m_tag in ['tuned_elasticnet', 'lgbm', 'xgb', 'catboost', 'dt', 'svr']:
            for _t in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
                key = f'{m_tag}_{_t}'
                if key not in oof_pred_store:
                    oof_pred_store[key] = {'y_true': [], 'y_pred': []}
        fold_records.append(fold_rec)

    return fold_records, oof_pred_store



def print_summary(fold_records, MAX_FOLDS):
    # Fold summary
    print()
    df_fold_summary = pd.DataFrame(fold_records)
    print("=== Fold Summary ===")
    # Show only metric columns here
    summary_cols = (['fold', 'held_out_pid']
                    + [c for c in df_fold_summary.columns if 'mae' in c or 'r2' in c])
    print(df_fold_summary[summary_cols].to_string(index=False))
    print()
    print(f"[OK] {len(fold_records)}/{MAX_FOLDS} folds completed")

    return df_fold_summary



def hyperparameter_frequency(df_fold_summary,TARGET_SIS, TARGET_OHS, TARGET_OKS):
    # Count how often each best-parameter setting was selected across folds

    TUNED_TAGS = ['tuned_ridge', 'tuned_elasticnet', 'rf', 'lgbm', 'xgb', 'catboost', 'dt', 'svr']
    TARGETS_ALL = [TARGET_SIS, TARGET_OHS, TARGET_OKS]

    print('=== Best-Param Frequency Summary (across 18 LOPO folds) ===\n')
    for model_tag in TUNED_TAGS:
        for target in TARGETS_ALL:
            col = f'{model_tag}_best_{target}'
            if col not in df_fold_summary.columns:
                continue
            vals = df_fold_summary[col].dropna().tolist()
            counts = Counter(vals)
            print(f'  [{model_tag.upper()}] target={target.upper()}')
            for param_str, cnt in counts.most_common():
                print(f'    {param_str}  ->  {cnt}/{len(vals)} folds')
            print()

    return 0



def run_delta_model(df_model_delta, ana, param_grid, logo_d, groups_d, MAX_FOLDS_D, WINDOWS, ANA_QUAL_COLS, PARTICIPANT_COL, ALL_NUM_COLS, FEAT_COLS_ROLLED, TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA, DATE_COL, SAFE_SENSOR_COLS, seed):

    DELTA_SPECS = [    ## TODO inserire i miei modelli in automatico
    # (tag, model_class, model_kwargs, param_grid, selector)
    ('tuned_ridge', Ridge, dict(), param_grid["PARAM_GRID_RIDGE"], None),
    ('tuned_elasticnet', ElasticNet, dict(max_iter=10000, random_state=seed), param_grid["PARAM_GRID_ELASTICNET"], None),
    ('rf', RandomForestRegressor, dict(random_state=seed, n_jobs=1), param_grid["PARAM_GRID_RF"], None),
    ('lgbm', LGBMRegressor, dict(random_state=seed, verbosity=-1, n_jobs=1), param_grid["PARAM_GRID_LGB"], None),
    ('xgb', XGBRegressor, dict(eval_metric="rmse", tree_method="hist", random_state=seed), param_grid["PARAM_GRID_XGBOOST"], None),
    ('catboost', CatBoostRegressor, dict(eval_metric="RMSE", verbose=False, random_seed=seed), param_grid["PARAM_GRID_CATBOOST"], None),
    ('dt', DecisionTreeRegressor, dict(random_state=seed), param_grid["PARAM_GRID_DECISIONTREE"], None),
    ('svr', SVR, dict(), param_grid["PARAM_GRID_SVR"], None)
    ]

    print(f"  Models   : {[s[0] for s in DELTA_SPECS]}")

    delta_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo_d.split(df_model_delta, groups=groups_d)):

        if fold_idx >= MAX_FOLDS_D:
            break

        #if fold_idx >= 1:  ### TODO: TEST
        #    break

        df_train_d = df_model_delta.iloc[train_idx].copy()
        df_test_d  = df_model_delta.iloc[test_idx].copy()

        train_pids_d   = set(df_train_d[PARTICIPANT_COL].unique())
        test_pids_d    = set(df_test_d[PARTICIPANT_COL].unique())
        held_out_pid_d = int(list(test_pids_d)[0])

        assert len(train_pids_d & test_pids_d) == 0, \
            f"Fold {fold_idx+1}: participant overlap in delta loop"

        # Merge demographics inside the fold
        df_train_dm = _merge_demographics(df_train_d, ana, PARTICIPANT_COL)
        df_test_dm  = _merge_demographics(df_test_d, ana, PARTICIPANT_COL)

        # Build rolling features separately for train and test
        df_tr_d_fe = pp.make_features(df_train_dm, WINDOWS, PARTICIPANT_COL, DATE_COL, SAFE_SENSOR_COLS)
        df_te_d_fe = pp.make_features(df_test_dm,  WINDOWS, PARTICIPANT_COL, DATE_COL, SAFE_SENSOR_COLS)

        n_inner_d   = min(len(train_pids_d), 5)
        inner_cv_d  = GroupKFold(n_splits=n_inner_d)
        inner_grp_d = df_tr_d_fe[PARTICIPANT_COL].values

        fold_rec_d = {
            'fold'         : fold_idx + 1,
            'held_out_pid' : held_out_pid_d,
        }

        print(f"  Fold {fold_idx+1:02d}/{MAX_FOLDS_D} | held-out P{held_out_pid_d:02d}")

        # Naive mean baseline for delta targets  (sarebbe quello che chiamavo Baseline B)
        for dtgt in [TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA]:
            train_delta_mean = float(df_train_dm[dtgt].mean())
            y_te_d_vals = df_test_dm[dtgt].values
            y_pred_d_bl = np.full(len(y_te_d_vals), train_delta_mean)
            mae_d_bl = mean_absolute_error(y_te_d_vals, y_pred_d_bl)
            r2_d_bl  = r2_score(y_te_d_vals, y_pred_d_bl)
            fold_rec_d[f'naive_mean_mae_{dtgt}'] = round(mae_d_bl, 4)
            fold_rec_d[f'naive_mean_r2_{dtgt}']  = round(r2_d_bl,  4)
            print(f"    -> [{dtgt}] naive_mean          "
                f"MAE={mae_d_bl:.3f}  R2={r2_d_bl:.3f}")

        for model_tag, model_cls, model_kwargs, model_grid, selector in DELTA_SPECS:
            for dtgt in [TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA]:
                estimator = model_cls(**model_kwargs)
                pipe = pp.build_model_pipeline(estimator, ALL_NUM_COLS)   ## ANA_QUAL_COLS
                search = GridSearchCV(
                    pipe, 
                    model_grid,
                    cv=inner_cv_d,
                    scoring='neg_mean_absolute_error',
                    refit=True, 
                    n_jobs=-1,
                )
                search.fit(
                    df_tr_d_fe[FEAT_COLS_ROLLED],
                    df_tr_d_fe[dtgt],
                    groups=inner_grp_d,
                )
                y_pred_d = search.predict(df_te_d_fe[FEAT_COLS_ROLLED])

                mae_d = mean_absolute_error(df_te_d_fe[dtgt], y_pred_d)
                r2_d  = r2_score(df_te_d_fe[dtgt], y_pred_d)
                fold_rec_d[f'{model_tag}_mae_{dtgt}']  = round(mae_d, 4)
                fold_rec_d[f'{model_tag}_r2_{dtgt}']   = round(r2_d,  4)
                print(f"    -> [{dtgt}] {model_tag:<18}  "
                    f"MAE={mae_d:.3f}  R2={r2_d:.3f}")

        delta_records.append(fold_rec_d)

    print()
    print(f"[OK] {len(delta_records)}/{MAX_FOLDS_D} delta folds completed")

    return delta_records
        
