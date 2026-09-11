# MAISON-ML

Research repository for analyzing longitudinal MAISON-LLF data and comparing machine learning models for clinical score estimation. The project combines daily features describing activity, heart rate, movement, position, and sleep with participant demographics.

The current configuration compares **five regression models for the `sis` and `oks` targets**, using nested **Leave-One-Participant-Out (LOPO)** cross-validation. The repository includes tabular datasets, two notebooks for data preparation and exploration, the experimental script, and previously exported results.

## Repository structure

```text
MAISON-ML/
├── README.md
├── LICENSE
├── .gitignore
├── Comparative_Analysis_v5.1.py
├── ETL.ipynb
├── EDA.ipynb
├── bash cmd.txt
├── log.txt
├── timing.log
├── dataset/
│   ├── maison-llf-features.csv
│   ├── maison-llf-demographics.csv
│   ├── maison-llf-feature-descriptions.xlsx
│   └── dataset_for_EDA.csv
└── results/
    └── run_v5.1/
        ├── results_regression.xlsx
        ├── summary_regression_stats.xlsx
        └── figures_regression/
            ├── boxplot_models_sis.png
            └── boxplot_models_oks.png
```

## Data

The dimensions below refer to the files currently included in the repository.

| File | Contents and purpose |
| --- | --- |
| [`maison-llf-features.csv`](dataset/maison-llf-features.csv) | 1,008 observations from 18 participants, with 56 records per participant. Contains identifiers, timestamps, clinical scores, and sensor features. |
| [`maison-llf-demographics.csv`](dataset/maison-llf-demographics.csv) | 18 rows, one per participant: sex, age, fracture/procedure type, relationship, education, employment, and ethnicity. |
| [`maison-llf-feature-descriptions.xlsx`](dataset/maison-llf-feature-descriptions.xlsx) | Descriptive dictionary of sensor features. |
| [`dataset_for_EDA.csv`](dataset/dataset_for_EDA.csv) | ETL output: 1,008 rows containing original features, encoded demographic variables, and clinical score categories. |

The join key is `participant`. The main variable groups are:

- **Time references:** `timestamp` and `clinical-timestamp`.
- **Clinical measures:** `sis`, `ohs`, `oks`, their individual items (`sis-*`, `ohs-*`, `oks-*`), `tug`, and `chairstand`.
- **Activity and movement:** `acceleration-*`, `motion-*`, and `step-*`.
- **Heart rate:** `heartrate-*`.
- **Position:** `position-*`.
- **Sleep:** `sleep-*`, including heart rate measurements during sleep.

## Notebooks and scripts

### `ETL.ipynb` — data preparation

This notebook reads the two source CSV files, encodes demographic variables, and merges them with the observations using a **right join** on `participant`, retaining participants present in the demographic table.

Transformations include:

- `sex_male`: `male → 1`, `female → 0`;
- `education_label`: ordinal encoding from secondary education (`0`) to doctorate degree (`3`);
- `work_part_time`: `retired → 0`, `employed part-time → 1`;
- one-hot encoding of `fracture-type` and `ethnicity`;
- discretization of `sis`, `ohs`, and `oks` into four quartile-based classes, named `SISS_Category_Q`, `OHSS_Category_Q`, and `OKSS_Category_Q`.

The result is written to `dataset/dataset_for_EDA.csv`. The notebook also constructs a numerical matrix `X`, but exports the full `data` table.

**Portability:** the notebook's input paths use the `.CSV` extension, whereas the included files use `.csv`. On case-sensitive filesystems, correct both paths before running the notebook.

### `EDA.ipynb` — exploratory data analysis

This notebook reads `dataset_for_EDA.csv` and analyzes a subset of predictors manually defined in `pred_col`. Comments explain some exclusions in terms of high correlations or the choice of a reference category.

The notebook:

1. prints variable pairs with an absolute correlation greater than `0.7` and less than `0.99999`;
2. displays a correlation heatmap;
3. computes the **Variance Inflation Factor (VIF)** for the selected predictors;
4. exports predictors, targets, and participant identifiers to `dataset/dataset_for_R.csv`, using `;` as the delimiter and `latin1` encoding.

`dataset_for_R.csv` is generated during execution and is not currently included. The instruction to save the heatmap as a PDF is commented out. Feature selection performed here **is not automatically transferred to the model comparison script**.

### `Comparative_Analysis_v5.1.py` — model comparison

This is the entry point for the experiments. It reads the two source CSV files directly and repeats demographic preprocessing and score discretization; running the notebooks first is not required.

The active configuration uses tabular regression models:

| Model | Hyperparameter grid |
| --- | --- |
| XGBoost | `n_estimators: [20, 50]`, `max_depth: [5, 10, 20, 50]`, `learning_rate: [0.001, 0.01]` |
| CatBoost | `iterations: [20, 50]`, `depth: [5, 10]`, `learning_rate: [0.001, 0.01]` |
| LightGBM | `n_estimators: [20, 50]`, `max_depth: [5, 10, 20, 50]`, `learning_rate: [0.001, 0.01]` |
| Decision Tree | `max_depth: [5, 10, 20, 50]` |
| Support Vector Regression (SVR) | `kernel: ["rbf"]`, `C: [1, 10]`, `epsilon: [0.01, 0.2]`, `gamma: ["scale"]` |

The configured random seed is `69`. The active targets are `sis` and `oks`; `ohs` is present in the data and preprocessing, but excluded from the regression target list.

#### Validation protocol

1. **Outer LOPO:** all observations from one participant form the test set; observations from the remaining participants form the training set.
2. **Inner LOPO:** for each hyperparameter combination, each participant in the outer training set is held out in turn for validation.
3. **Selection:** the combination with the highest mean inner-fold `R²` is selected, with equal weight assigned to each fold.
4. **Refitting:** the selected model is trained on the entire outer training set.
5. **Evaluation:** metrics are computed for the held-out participant, and both metrics and selected hyperparameters are saved.

The current dataset yields 18 outer folds and 17 inner folds per outer fold. A complete run without errors is expected to produce **90 rows per target**: 18 participants × 5 models. The search requires many model fits and can take substantial time.

#### Additional functions and components

| Component | Purpose and status |
| --- | --- |
| `inner_LoPo_ParOpt` | Exhaustive search with inner LOPO; used by the active workflow. |
| `inner_GridSearch_ParOpt` | Inactive alternative using `GridSearchCV` and `LeaveOneGroupOut`. Scoring is fixed to `f1_macro`, so it cannot be used directly for regression. |
| `create_sequences` | Builds sliding sequences; unused in the current comparison. |
| `build_lstm_model`, `build_lstm_model_OLD`, `build_tcn_model` | Unused experimental functions, with required imports commented out and unresolved references. |
| `save_ConfMat` | Plots a four-class confusion matrix; not called by the current workflow. |

### `cmd.txt` and logs

`cmd.txt` collects working commands for activating the `ml_env` Conda environment, running the script with output redirection, checking Python processes on Windows, and exporting dependencies with `pip freeze`. It is an operational reference.

- `log.txt` captures stdout and stderr when output redirection is used; the included file records a Control-C interruption.
- `timing.log` records the model, target, and elapsed time for inner search, refitting, and evaluation for each completed combination within an outer fold. It does not include the participant identifier and is **overwritten at every startup**.

## Installation and execution

No dependency file with pinned versions is included. The commands below install packages identified from the source imports; they do not reconstruct the original experimental environment.

### 1. Prepare the environment

Open a terminal at the repository root and create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install numpy pandas scipy scikit-learn xgboost catboost lightgbm matplotlib seaborn openpyxl gensim numba sktime statsmodels jupyterlab
```

`gensim`, `numba`, and `sktime` are imported by the script even though their components are unused in the active comparison. `statsmodels` is required for VIF computation in the EDA notebook. TensorFlow, SciKeras, and TCN are not required by the current execution path.

### 2. Run the experiments

Paths are relative to the working directory: run commands from the **repository root**. Ensure the results directory exists:

```bash
python -c "from pathlib import Path; Path('results/run_v5.1').mkdir(parents=True, exist_ok=True)"
python Comparative_Analysis_v5.1.py > log.txt 2>&1
```

The script writes `results/run_v5.1/results_regression.xlsx`, overwriting the existing file. Preserve previous results separately or change `output_path` before starting a new run. The command above also overwrites `log.txt`.

For a new scientific evaluation, first address the predictor selection issue described under limitations. This command executes the current code as provided.

### 3. Run data preparation and exploration

```bash
jupyter lab
```

Open and execute the cells of `ETL.ipynb` in order, followed by those of `EDA.ipynb`. If using the included `dataset_for_EDA.csv`, EDA can be opened directly. Notebook exports overwrite their respective CSV files.

### 4. Change the configuration

The script does not expose a command-line interface. Settings are edited in the source:

| Variable | Setting |
| --- | --- |
| `seed` | Random seed, currently `69`. |
| `reg_responses` | Regression targets, currently `["sis", "oks"]`. |
| `TABULAR_MODELS` | Estimators and search grids. |
| `inner_ParOpt` | Active optimization strategies, currently LOPO only. |
| `exclude_cols`, `feature_cols`, `X` | Predictor selection, which must be checked against the targets. |
| `output_path` | Destination of the Excel output. |

## Results and interpretation

[`results_regression.xlsx`](results/run_v5.1/results_regression.xlsx) contains one sheet per target (`sis`, `oks`). Each row generated by the script describes a model evaluated on a held-out participant:

| Column | Meaning |
| --- | --- |
| `Patient` | Participant excluded from outer training. |
| `Model` | Evaluated model. |
| `CV` | Inner optimization strategy (`LOPO`). |
| `MAE`, `MSE`, `RMSE` | Mean absolute error, mean squared error, and root mean squared error. |
| `R2` | Coefficient of determination on the outer test fold. |
| `MAPE` | Mean absolute percentage error returned as a ratio: `0.1` corresponds to 10%. |
| `MEDAE` | Median absolute error. |
| `EVS` | Explained variance score. |
| `Parameters` | Hyperparameters selected in the fold, serialized as JSON. |

The repository also includes [`summary_regression_stats.xlsx`](results/run_v5.1/summary_regression_stats.xlsx), with `sis` and `oks` sheets, and two boxplots in [`figures_regression`](results/run_v5.1/figures_regression). **The code generating this summary and these figures is not included**. Trained models and individual predictions are not saved.

## Limitations and reproducibility

- **Target leakage in the inputs.** `X` retains numerical columns after excluding the specified identifiers, timestamps, and score categories. Consequently, `sis`, `ohs`, `oks`, and their individual items remain available as predictors. Before a new evaluation, define the inputs available in the intended prediction setting, exclude the target and variables that determine or indirectly reveal it, and rerun the comparison. Separating participants does not eliminate this issue.
- **Global discretization.** Quartiles are computed on the entire dataset, even when running regression only. If classification is reactivated, thresholds must be defined without using the test set. Duplicate quartile boundaries or missing values can also interrupt preprocessing.
- **Limited preprocessing.** Imputation and scaling are not implemented. Categories absent from the demographic mappings produce missing values. Any transformations learned from data must be fitted within the training folds.
- **Errors and result completeness.** Exceptions during model comparison are printed and execution continues, so an Excel file may contain fewer rows than expected. Check the logs and confirm that every participant is represented for each model and target. Warnings are globally suppressed in the script.
- **Longitudinal evaluation.** LOPO separates participants but does not implement forecasting of future observations. The current configuration treats observations as tabular rows.
- **Unpinned environment.** Dependency versions and the original environment specification are unavailable. The seed does not guarantee identical results across platforms and package versions. For new runs, record the Git revision, Python version, dependencies, and experimental configuration; `python -m pip freeze > requirements.txt` exports the environment used.

## License and scientific references

The repository includes the text of the **GNU General Public License, version 3**, in [`LICENSE`](LICENSE).

The inspected files do not provide a formal bibliographic citation for the project or dataset. To cite this implementation, report the repository URL and the revision used. References to the associated publication and dataset provenance should be added using official project information.
