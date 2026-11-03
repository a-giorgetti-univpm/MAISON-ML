import pandas as pd
import numpy as np



def save_baseline(fold_records, TARGET_SIS, TARGET_OHS, TARGET_OKS, TABLES_DIR):

    MODEL_PREFIX_MAP = {
    'baseline_demo_ridge' : 'bl_a_',
    'baseline_naive_mean' : 'bl_b_',
    }

    baseline_rows = []
    for rec in fold_records:
        for model_name, prefix in MODEL_PREFIX_MAP.items():
            for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
                baseline_rows.append({
                    'model'        : model_name,
                    'target'       : target,
                    'fold'         : rec['fold'],
                    'held_out_pid' : rec['held_out_pid'],
                    'n_train_rows' : rec['n_train_rows'],
                    'n_test_rows'  : rec['n_test_rows'],
                    'mae'          : rec.get(f'{prefix}mae_{target}', np.nan),  ## rec.get(chiave, np.nan) legge un valore dal dizionario rec. Se la chiave c'è, restituisce il valore; se manca, restituisce np.nan invece di sollevare un KeyError
                    'r2'           : rec.get(f'{prefix}r2_{target}',  np.nan),
                })

    df_baselines = pd.DataFrame(baseline_rows)

    # Save to CSV
    out_path = TABLES_DIR / 'baselines.csv'
    df_baselines.to_csv(out_path, index=False)

    # Per-fold table
    print("=== Per-fold baseline results ===")
    print(df_baselines.to_string(index=False))
    print()

    # Summary: mean +/- std across folds
    print("=== Baseline Summary (mean +/- std across folds) ===")
    grp = (df_baselines
        .groupby(['model', 'target'])[['mae', 'r2']]
        .agg(['mean', 'std'])      ### media tra le 18 fold
        .round(4))
    grp.columns = ['mae_mean', 'mae_std', 'r2_mean', 'r2_std']
    print(grp.to_string())
    print()

    # Basic checks
    assert out_path.exists(), f"baselines.csv not found at {out_path}"
    assert 'mae' in df_baselines.columns and 'r2' in df_baselines.columns
    assert set(df_baselines['target'].unique()) == {TARGET_SIS, TARGET_OHS, TARGET_OKS}, \
        "Missing SIS or OHS in baselines.csv"
    assert df_baselines['mae'].notna().all(), "NaN MAE values in baselines.csv"

    n_models = df_baselines['model'].nunique()
    n_targets = df_baselines['target'].nunique()
    n_folds_out = df_baselines['fold'].nunique()
    print(f"[OK] Saved -> {out_path}")
    print(f"[OK] {n_models} models x {n_targets} targets x {n_folds_out} folds"
        f" = {len(df_baselines)} rows")

    return 0



def save_long_table(fold_records, TARGET_SIS, TARGET_OHS, TARGET_OKS, TABLES_DIR):

    ALL_MODEL_MAP = {
    'baseline_demo_ridge' : 'bl_a_',
    'baseline_naive_mean' : 'bl_b_',
    'ridge_sensor_demo'   : 'ridge_',
    'tuned_ridge'         : 'tuned_ridge_',
    'tuned_elasticnet'    : 'tuned_elasticnet_',
    'rf'                  : 'rf_',
    'lgbm'                : 'lgbm_',
    }

    metrics_rows = []
    for rec in fold_records:
        for model_name, prefix in ALL_MODEL_MAP.items():   ## items() restituisce le coppie (chiave, valore) del dizionario
            for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
                mae_key = f'{prefix}mae_{target}'
                r2_key  = f'{prefix}r2_{target}'
                if mae_key not in rec:
                    continue    # skip if model not run in this fold
                metrics_rows.append({
                    'model'        : model_name,
                    'target'       : target,
                    'fold'         : rec['fold'],
                    'held_out_pid' : rec['held_out_pid'],
                    'n_train_rows' : rec['n_train_rows'],
                    'n_test_rows'  : rec['n_test_rows'],
                    'mae'          : rec[mae_key],
                    'r2'           : rec[r2_key],
                })

    df_all_metrics = pd.DataFrame(metrics_rows)

    # Save per-fold results
    per_part_path = TABLES_DIR / 'lopo_metrics_per_participant.csv'
    df_all_metrics.to_csv(per_part_path, index=False)

    # Save summary across folds
    summary = (df_all_metrics
            .groupby(['model', 'target'])[['mae', 'r2']]
            .agg(['mean', 'std'])
            .round(4))
    summary.columns = ['mae_mean', 'mae_std', 'r2_mean', 'r2_std']
    summary = summary.reset_index()
    summary_path = TABLES_DIR / 'lopo_metrics_summary.csv'
    summary.to_csv(summary_path, index=False)

    # Print summary table
    print("=== All Model Results -- Summary (mean +/- std across LOPO folds) ===")
    print()
    for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
        print(f"  Target: {target.upper()}")
        sub = summary[summary['target'] == target].copy()
        sub['MAE'] = sub.apply(
            lambda r: f"{r['mae_mean']:.3f} +/- {r['mae_std']:.3f}", axis=1)
        ## sub è la tabella riassuntiva di un solo target, per esempio sis: una riga per modello, 
        ## con mae_mean e mae_std già calcolati sulle fold. 
        ## apply(..., axis=1) scorre le righe. r è la riga corrente, e la lambda legge i due numeri 
        ## e li scrive in una stringa a tre decimali. Se la media è 4.21678 e la std è 1.20491, 
        ## la cella diventa "4.217 +/- 1.205"
        sub['R2'] = sub.apply(
            lambda r: f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f}", axis=1)
        print(sub[['model', 'MAE', 'R2']].to_string(index=False))
        print()

    # Basic checks
    assert per_part_path.exists(),  "lopo_metrics_per_participant.csv not saved"
    assert summary_path.exists(),   "lopo_metrics_summary.csv not saved"
    assert 'mae' in df_all_metrics.columns and 'r2' in df_all_metrics.columns

    #STEP7_MODELS    = {'tuned_ridge', 'tuned_elasticnet', 'rf', 'lgbm'}
    #tuned_present   = STEP7_MODELS & set(df_all_metrics['model'].unique())
    #missing_step7   = STEP7_MODELS - set(df_all_metrics['model'].unique())
    #assert len(tuned_present) == 4, f"LOPO models missing from metrics: {missing_step7}"

    n_models  = df_all_metrics['model'].nunique()
    n_targets = df_all_metrics['target'].nunique()
    n_f       = df_all_metrics['fold'].nunique()
    print(f"[OK] Saved -> {per_part_path}")
    print(f"[OK] Saved -> {summary_path}")
    print(f"[OK] {n_models} models x {n_targets} targets x {n_f} folds = {len(df_all_metrics)} rows")

    return df_all_metrics, summary



def save_delta_summary(delta_records, TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA, TABLES_DIR):

    DELTA_MODEL_MAP = {
    'baseline_naive_mean' : 'naive_mean_',
    'tuned_ridge'         : 'tuned_ridge_',
    'tuned_elasticnet'    : 'tuned_elasticnet_',
    'rf'                  : 'rf_',
    'lgbm'                : 'lgbm_',
    }

    delta_metric_rows = []
    for rec in delta_records:
        for model_name, prefix in DELTA_MODEL_MAP.items():
            for dtgt in [TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA]:
                mae_key = f'{prefix}mae_{dtgt}'
                r2_key  = f'{prefix}r2_{dtgt}'
                if mae_key not in rec:
                    continue
                delta_metric_rows.append({
                    'model'        : model_name,
                    'target'       : dtgt,
                    'fold'         : rec['fold'],
                    'held_out_pid' : rec['held_out_pid'],
                    'mae'          : rec[mae_key],
                    'r2'           : rec[r2_key],
                })

    df_delta_metrics = pd.DataFrame(delta_metric_rows)

    # Save per-fold results
    delta_per_path = TABLES_DIR / 'lopo_metrics_per_participant_delta.csv'
    df_delta_metrics.to_csv(delta_per_path, index=False)

    # Save summary across folds
    delta_summary = (
        df_delta_metrics
        .groupby(['model', 'target'])[['mae', 'r2']]
        .agg(['mean', 'std'])
        .round(4)
    )
    delta_summary.columns = ['mae_mean', 'mae_std', 'r2_mean', 'r2_std']
    delta_summary = delta_summary.reset_index()
    delta_sum_path = TABLES_DIR / 'lopo_metrics_summary_delta.csv'
    delta_summary.to_csv(delta_sum_path, index=False)

    # Print summary
    print("=== Delta-Target Results (mean +/- std across LOPO folds) ===")
    for dtgt in [TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA]:
        print(f"\n  Target: {dtgt.upper()}")
        _sub = delta_summary[delta_summary['target'] == dtgt].copy()
        _sub['MAE'] = _sub.apply(
            lambda r: f"{r['mae_mean']:.3f} +/- {r['mae_std']:.3f}", axis=1)
        _sub['R2']  = _sub.apply(
            lambda r: f"{r['r2_mean']:.3f} +/- {r['r2_std']:.3f}", axis=1)
        print(_sub[['model', 'MAE', 'R2']].to_string(index=False))

    return df_delta_metrics, delta_summary, delta_per_path, delta_sum_path




def check_delta_outputs_saved(delta_per_path, delta_sum_path, FIGURES_DIR):
    
    # Check that the delta outputs were saved
    for _p in [delta_per_path, delta_sum_path,
            FIGURES_DIR / 'model_comparison_sis_delta.png',
            FIGURES_DIR / 'model_comparison_ohs_delta.png',
            FIGURES_DIR / 'model_comparison_oks_delta.png']:
        assert _p.exists() and _p.stat().st_size > 0, f"Missing delta output: {_p}"
        print(f'[OK] {_p.name}: {_p.stat().st_size:,} bytes')

    print()
    print('[OK] All delta outputs saved.')

    return 0