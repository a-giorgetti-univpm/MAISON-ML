import matplotlib.pyplot as plt
import numpy as np



def _make_comparison_fig(summary, target, out_path, df_all_metrics, MODEL_ORDER, MODEL_LABELS, PALETTE):
        
        sub = (summary[summary['target'] == target]
            .set_index('model')
            .reindex(MODEL_ORDER)   ### riordina le righe di sub secondo la lista MODEL_ORDER
            .dropna(subset=['mae_mean'])
            .reset_index())
        labels = [MODEL_LABELS[m] for m in sub['model']]
        colors = [PALETTE[MODEL_ORDER.index(m)] for m in sub['model']]
        x      = np.arange(len(sub))

        n_folds_run = df_all_metrics['fold'].nunique()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(
            f'Model Comparison -- {target.upper()}  '
            f'(LOPO-CV, mean +/- std, {n_folds_run} folds)',
            fontsize=11, fontweight='bold',
        )

        metrics_spec = [
            (axes[0], 'mae_mean', 'mae_std', 'Mean Absolute Error', 'MAE (lower is better)'),
            (axes[1], 'r2_mean',  'r2_std',  'R2 Score',            'R2 (higher is better)'),
        ]
        for ax, metric, metric_std, title, ylabel in metrics_spec:
            std_vals = sub[metric_std].fillna(0)
            bars = ax.bar(
                x, sub[metric], width=0.55,
                yerr=std_vals, capsize=4,
                color=colors, edgecolor='white', linewidth=0.5,
                error_kw=dict(ecolor='#555555', elinewidth=1.2, capthick=1.2),
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            if metric == 'r2_mean':
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

            # Add value labels above the bars
            for bar, val, err in zip(bars, sub[metric], std_vals):
                label_y = val + err + abs(val) * 0.03 + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5,
                )

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[OK] Saved -> {out_path}')



def make_model_comparison_figures(df_all_metrics, summary, TARGET_SIS, TARGET_OHS, TARGET_OKS, FIGURES_DIR):
    
    MODEL_ORDER = [                 ## TODO: aggiungere i miei modelli in automatico
        'baseline_naive_mean',
        'baseline_demo_ridge',
        'ridge_sensor_demo',
        'tuned_ridge',
        'tuned_elasticnet',
        'rf',
        'lgbm',
        'xgb',
        'catboost',
        'dt',
        'svr'
    ]
    
    MODEL_LABELS = {                ## TODO: aggiungere i miei modelli in automatico
        'baseline_naive_mean' : 'Naive\nMean',
        'baseline_demo_ridge' : 'Demo\nRidge',
        'ridge_sensor_demo'   : 'Ridge\n(untuned)',
        'tuned_ridge'         : 'Ridge\n(tuned)',
        'tuned_elasticnet'    : 'Elastic\nNet',
        'rf'                  : 'Random\nForest',
        'lgbm'                : 'LightGBM',
        'xgb'                 : 'XGBoost',
        'catboost'            : 'CatBoost',
        'dt'                  : 'Decision\nTree',
        'svr'                 : 'Support\nVector Regression'
    }

    # Baselines in red, sensor models in blue/green/orange
    PALETTE = ['#c0392b', '#e07b54', '#5b9bd5', '#2980b9', '#16a085', '#27ae60', '#e67e22', '#8e44ad', '#f39c12', '#d35400', '#7f8c8d'] 

    _make_comparison_fig(summary, TARGET_SIS, FIGURES_DIR / 'model_comparison_sis.png', df_all_metrics, MODEL_ORDER, MODEL_LABELS, PALETTE)
    _make_comparison_fig(summary, TARGET_OHS, FIGURES_DIR / 'model_comparison_ohs.png', df_all_metrics, MODEL_ORDER, MODEL_LABELS, PALETTE)
    _make_comparison_fig(summary, TARGET_OKS, FIGURES_DIR / 'model_comparison_oks.png', df_all_metrics, MODEL_ORDER, MODEL_LABELS, PALETTE)
   

    # Check that the figures were saved
    for fname in ['model_comparison_sis.png', 'model_comparison_ohs.png', 'model_comparison_oks.png']:
        p = FIGURES_DIR / fname
        assert p.exists(), f"Missing figure: {fname}"
        print(f'[OK] {fname}: {p.stat().st_size:,} bytes')

    print('[OK] Model comparison figures complete')

    return 0



def plot_actual_predicted(oof_pred_store, TARGET_SIS, TARGET_OHS, TARGET_OKS, FIGURES_DIR):     ### TODO: valutare altri modelli per gli scatter
    
    # Plot pooled out-of-fold predictions for ElasticNet and LightGBM

    SCATTER_MODELS = {
        'tuned_elasticnet': 'ElasticNet (tuned)',
        'lgbm':             'LightGBM',
    }
    TARGET_RANGES = {
        TARGET_SIS: (15, 30, 'SIS'),
        TARGET_OHS: (0,  48, 'OHS'),
        TARGET_OKS: (0,  48, 'OKS'),
    }

    fig, axes = plt.subplots(len(SCATTER_MODELS), len(TARGET_RANGES),
                            figsize=(13, 8))
    fig.suptitle('Actual vs Predicted — Pooled OOF (all 18 LOPO folds)',
                fontsize=11, fontweight='bold')

    for row_i, (m_tag, m_label) in enumerate(SCATTER_MODELS.items()):
        for col_i, (target, (tmin, tmax, tlabel)) in enumerate(TARGET_RANGES.items()):
            ax = axes[row_i][col_i]
            key = f'{m_tag}_{target}'
            if key not in oof_pred_store or not oof_pred_store[key]['y_true']:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            y_true = np.array(oof_pred_store[key]['y_true'])
            y_pred_oof = np.array(oof_pred_store[key]['y_pred'])
            ax.scatter(y_true, y_pred_oof, alpha=0.25, s=8,
                    color='#2980b9', edgecolors='none')
            diag = np.linspace(tmin, tmax, 100)
            ax.plot(diag, diag, 'r--', linewidth=1, alpha=0.7, label='perfect')
            mae_all = np.mean(np.abs(y_true - y_pred_oof))
            ax.set_xlabel(f'Actual {tlabel}', fontsize=8)
            ax.set_ylabel(f'Predicted {tlabel}', fontsize=8)
            ax.set_title(f'{m_label} — {tlabel}\nMAE={mae_all:.3f}', fontsize=9)
            ax.set_xlim(tmin - 2, tmax + 2)
            ax.set_ylim(tmin - 2, tmax + 2)
            ax.set_aspect('equal', 'box')
            ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    scatter_path = FIGURES_DIR / 'oof_scatter_actual_vs_predicted.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    #plt.show()
    plt.close('all')
    print(f'[OK] Saved -> {scatter_path}')

    return 0



def _make_delta_fig(delta_summary, df_delta_metrics, dtgt, out_path, DELTA_MODEL_ORDER, DELTA_MODEL_LABELS, DELTA_PALETTE):
        _sub = (
            delta_summary[delta_summary['target'] == dtgt]
            .set_index('model')
            .reindex(DELTA_MODEL_ORDER)
            .dropna(subset=['mae_mean'])
            .reset_index()
        )
        labels = [DELTA_MODEL_LABELS[m] for m in _sub['model']]
        colors = [DELTA_PALETTE[DELTA_MODEL_ORDER.index(m)] for m in _sub['model']]
        x = np.arange(len(_sub))
        n_folds_run_d = df_delta_metrics['fold'].nunique()

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.suptitle(
            f'Delta-Target Comparison -- {dtgt.upper()}  '
            f'(LOPO-CV, {n_folds_run_d} folds)',
            fontsize=11, fontweight='bold',
        )
        metrics_spec = [
            (axes[0], 'mae_mean', 'mae_std', 'Mean Absolute Error', 'MAE (lower is better)'),
            (axes[1], 'r2_mean',  'r2_std',  'R2 Score',            'R2 (higher is better)'),
        ]
        for ax, metric, metric_std, title, ylabel in metrics_spec:
            std_vals = _sub[metric_std].fillna(0)
            bars = ax.bar(
                x, _sub[metric], width=0.45,
                yerr=std_vals, capsize=4,
                color=colors, edgecolor='white', linewidth=0.5,
                error_kw=dict(ecolor='#555555', elinewidth=1.2, capthick=1.2),
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            if metric == 'r2_mean':
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            for bar, val, err in zip(bars, _sub[metric], std_vals):
                label_y = val + err + abs(val) * 0.03 + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2, label_y,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[OK] Saved -> {out_path}')



def make_delta_comparison_figures(delta_summary, df_delta_metrics, TARGET_SIS_DELTA, TARGET_OHS_DELTA, TARGET_OKS_DELTA, FIGURES_DIR):

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

    # Delta comparison figures    ## TODO: aggiungere i miei modelli in automatico
    DELTA_MODEL_ORDER  = ['tuned_ridge', 'tuned_elasticnet', 'rf', 'lgbm', 'xgb', 'catboost', 'dt', 'svr']
    DELTA_MODEL_LABELS = {
        'tuned_ridge'      : 'Ridge\n(tuned)',
        'tuned_elasticnet' : 'Elastic\nNet',
        'rf'               : 'Random\nForest',
        'lgbm'             : 'LightGBM',
        'xgb'              : 'XGBoost',
        'catboost'         : 'CatBoost',
        'dt'               : 'Decision\nTree',
        'svr'              : 'Support\nVector Regression'
    }
    DELTA_PALETTE = ['#2980b9', '#16a085', '#27ae60', '#e67e22', '#8e44ad', '#f39c12', '#d35400', '#7f8c8d']

    _make_delta_fig(delta_summary, df_delta_metrics, TARGET_SIS_DELTA, FIGURES_DIR / 'model_comparison_sis_delta.png', DELTA_MODEL_ORDER, DELTA_MODEL_LABELS, DELTA_PALETTE)
    _make_delta_fig(delta_summary, df_delta_metrics, TARGET_OHS_DELTA, FIGURES_DIR / 'model_comparison_ohs_delta.png', DELTA_MODEL_ORDER, DELTA_MODEL_LABELS, DELTA_PALETTE)
    _make_delta_fig(delta_summary, df_delta_metrics, TARGET_OKS_DELTA, FIGURES_DIR / 'model_comparison_oks_delta.png', DELTA_MODEL_ORDER, DELTA_MODEL_LABELS, DELTA_PALETTE)


    return 0