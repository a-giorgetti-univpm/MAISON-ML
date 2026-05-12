import os
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

# Define root directory
root = '.'

# Load dataset
(samples, siss, ohss, okss, participants) = torch.load(os.path.join(root, 'dataset/dataset-daily.pt'))

# Convert tensors to pandas DataFrame
df = pd.DataFrame({
    "SISS": siss,
    "OHSS": ohss,
    "OKSS": okss
})

# Compute quartiles for discretization
siss_q1, siss_q2, siss_q3 = np.percentile(df["SISS"], [25, 50, 75])
ohss_q1, ohss_q2, ohss_q3 = np.percentile(df["OHSS"], [25, 50, 75])
okss_q1, okss_q2, okss_q3 = np.percentile(df["OKSS"], [25, 50, 75])

# Define quartile-based bins and labels
quartile_labels = [0, 1, 2, 3]

# Apply discretization
df["SISS_Category_Q"] = pd.cut(df["SISS"], bins=[df["SISS"].min(), siss_q1, siss_q2, siss_q3, df["SISS"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)
df["OHSS_Category_Q"] = pd.cut(df["OHSS"], bins=[df["OHSS"].min(), ohss_q1, ohss_q2, ohss_q3, df["OHSS"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)

df["OKSS_Category_Q"] = pd.cut(df["OKSS"], bins=[df["OKSS"].min(), okss_q1, okss_q2, okss_q3, df["OKSS"].max()],
                               labels=quartile_labels, include_lowest=True).astype(int)


# Extract features, target, and patient IDs for LOPO
X = pd.DataFrame(samples)
y = df["OHSS_Category_Q"]
groups = pd.Series(participants)

# Define classifier and hyperparameter grid
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
param_grid = {"max_depth": [3, 5, 7], "n_estimators": [20, 50]}

# Leave-One-Patient-Out CV
outer_logo = LeaveOneGroupOut()

# Initialize results storage
overall_conf_matrix = np.zeros((4, 4))  # Assuming 4 categories (0, 1, 2, 3)
performance_metrics = []

# Outer LOPO Loop
count=0
for train_idx, test_idx in outer_logo.split(X, y, groups):
    count=count+1
    print(count)
    X_train_outer, X_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
    y_train_outer, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
    groups_train_outer = groups.iloc[train_idx]

    # Inner LOPO for Hyperparameter Optimization
    inner_logo = LeaveOneGroupOut()
    best_model = None
    best_score = -np.inf

    for inner_train_idx, inner_val_idx in inner_logo.split(X_train_outer, y_train_outer, groups_train_outer):
        X_train_inner, X_val = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
        y_train_inner, y_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

        # Hyperparameter tuning
        for params in ParameterGrid(param_grid):
            params = {k: int(v) if isinstance(v, np.generic) else v for k, v in params.items()}
            model.set_params(**params)
            model.fit(X_train_inner, y_train_inner)
            y_val_pred = model.predict(X_val)
            score = f1_score(y_val, y_val_pred, average="macro")

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params  # Store the best hyperparameters

    # Train best model on full outer training set
    best_model.fit(X_train_outer, y_train_outer)
    y_pred = best_model.predict(X_test)

    # Compute metrics
    f1 = f1_score(y_test, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    # Aggregate confusion matrices
    overall_conf_matrix += conf_matrix

    # Store results
    performance_metrics.append([f1, balanced_acc, recall, precision])

# Convert results to DataFrame
performance_df = pd.DataFrame(performance_metrics, columns=["Macro-F1", "Balanced Accuracy", "Macro Recall", "Macro Precision"])

# Save performance metrics and overall confusion matrix
output_path = os.path.join(root, "results/model_performance_ohss.xlsx")
with pd.ExcelWriter(output_path) as writer:
    performance_df.to_excel(writer, sheet_name="All_Folds")

# Plot overall confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(overall_conf_matrix, annot=True, cmap="coolwarm", xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix OHSS")
conf_matrix_path = os.path.join(root, "results/overall_confusion_matrix_ohss.pdf")
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Performance metrics saved as: {output_path}")
print(f"✅ Overall Confusion Matrix saved as: {conf_matrix_path}")