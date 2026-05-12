import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Essential libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor
import torch

# Set random seed for reproducibility

seed = 69
torch.manual_seed(seed)
np.random.seed(seed)

# Define root directory
root = '.'

# Load dataset
(samples, siss, ohss, okss, participants) = torch.load(os.path.join(root, 'dataset/dataset-' + 'daily' + '.pt'))

# Display unique participants and their counts
unique_values, counts = np.unique(participants, return_counts=True)
print(unique_values)
print(counts)

# Initialize variables
x = samples
y = siss
p = participants

Y_TRUES = np.empty([0])
Y_PREDS = np.empty([0])

# Perform cross-validation
cv = KFold(n_splits=x.shape[0], shuffle=True, random_state=seed)
for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
    participant = np.unique(p[test_idx])[0]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(participant, x_train.shape[0], x_test.shape[0])

    # Standardize features
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Normalize features
    normalizer = MinMaxScaler()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)

    # Train model (CatBoostRegressor)
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=.1,
        depth=3,
        loss_function='RMSE',
        verbose=10000
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        use_best_model=True,
        early_stopping_rounds=100
    )
    y_preds = model.predict(x_test)

    # Append results
    Y_TRUES = np.append(Y_TRUES, y_test)
    Y_PREDS = np.append(Y_PREDS, y_preds)

# Sort results for plotting
indx = Y_TRUES.argsort()
Y_TRUES = Y_TRUES[indx]
Y_PREDS = Y_PREDS[indx]

# Plot results
plt.figure()
plt.plot(Y_TRUES, 'o', color='blue', alpha=.25, markersize=8, label='Ground-truth')
plt.plot(Y_PREDS, 'o', color='red', alpha=.25, markersize=8, label='Prediction')
plt.title('MAE = ' + str(mean_absolute_error(Y_TRUES, Y_PREDS).__round__(2)) +
          ', MSE = ' + str(mean_squared_error(Y_TRUES, Y_PREDS).__round__(2)) +
          ', R2 = ' + str(r2_score(Y_TRUES, Y_PREDS).__round__(2)) +
          ', Corr = ' + str(spearmanr(Y_TRUES, Y_PREDS)[0].__round__(2)))
plt.legend()
plt.show()

# Print performance metrics
print('MAE = ' + str(mean_absolute_error(Y_TRUES, Y_PREDS).__round__(2)) +
      ', MSE = ' + str(mean_squared_error(Y_TRUES, Y_PREDS).__round__(2)) +
      ', R2 = ' + str(r2_score(Y_TRUES, Y_PREDS).__round__(2)) +
      ', Corr = ' + str(spearmanr(Y_TRUES, Y_PREDS)[0].__round__(2)))
