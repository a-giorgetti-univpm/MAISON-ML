
# This version reads the saved CSV files, so it works after kernel restart.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLES_DIR = Path("outputs/run_v8/tables")
FIGURES_DIR = Path("outputs/run_v8/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

summary_path = TABLES_DIR / "lopo_metrics_summary.csv"
delta_summary_path = TABLES_DIR / "lopo_metrics_summary_delta.csv"

if not summary_path.exists():
    raise FileNotFoundError(f"Missing file: {summary_path}")

if not delta_summary_path.exists():
    raise FileNotFoundError(f"Missing file: {delta_summary_path}")

summary = pd.read_csv(summary_path)
delta_summary = pd.read_csv(delta_summary_path)

TARGET_ORDER_ABS = ["sis", "ohs", "oks"]
TARGET_ORDER_DELTA = ["sis_delta", "ohs_delta", "oks_delta"]

TARGET_LABELS = {
    "sis": "SIS",
    "ohs": "OHS",
    "oks": "OKS",
}

# ---------------------------------------------------
# Figure 1: Absolute-score MAE across main models
# ---------------------------------------------------
ABS_MODEL_ORDER = [
    "baseline_naive_mean",
    "tuned_elasticnet",
    "rf",
    "lgbm",
]

ABS_MODEL_LABELS = {
    "baseline_naive_mean": "Naive Mean",
    "tuned_elasticnet": "ElasticNet",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
}

x = np.arange(len(TARGET_ORDER_ABS))
width = 0.18

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, model in enumerate(ABS_MODEL_ORDER):
    values = []
    errors = []

    for target in TARGET_ORDER_ABS:
        row = summary[
            (summary["model"] == model) &
            (summary["target"] == target)
        ]

        if row.empty:
            values.append(np.nan)
            errors.append(0)
        else:
            values.append(float(row["mae_mean"].iloc[0]))
            errors.append(float(row["mae_std"].iloc[0]))

    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        yerr=errors,
        capsize=3,
        label=ABS_MODEL_LABELS[model],
    )

ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER_ABS])
ax.set_ylabel("MAE")
ax.set_xlabel("Clinical target")
ax.set_title("Absolute-score LOPO-CV MAE")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()

out_path = FIGURES_DIR / "absolute_mae_summary.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved -> {out_path}")

# ---------------------------------------------------
# Figure 2: Absolute-score vs delta-target MAE
# ---------------------------------------------------
COMPARE_SERIES = [
    ("baseline_naive_mean", "absolute", "Naive Mean Absolute"),
    ("baseline_naive_mean", "delta",    "Naive Mean Delta"),
    ("tuned_elasticnet",    "absolute", "ElasticNet Absolute"),
    ("tuned_elasticnet",    "delta",    "ElasticNet Delta"),
]

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, (model, setting, label) in enumerate(COMPARE_SERIES):
    values = []
    errors = []
    print(model)

    for abs_target, delta_target in zip(TARGET_ORDER_ABS, TARGET_ORDER_DELTA):
        if setting == "absolute":
            src = summary
            target = abs_target
        else:
            src = delta_summary
            target = delta_target

        row = src[
            (src["model"] == model) &
            (src["target"] == target)
        ]

        if row.empty:
            values.append(np.nan)
            errors.append(0)
        else:
            values.append(float(row["mae_mean"].iloc[0]))
            errors.append(float(row["mae_std"].iloc[0]))

    ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        yerr=errors,
        capsize=3,
        label=label,
    )

    ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER_ABS])
ax.set_ylabel("MAE")
ax.set_xlabel("Clinical target")
ax.set_title("Absolute-score vs delta-target MAE")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()

out_path = FIGURES_DIR / "absolute_vs_delta_mae.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved -> {out_path}")