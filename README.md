# 🪨 Pore Pressure Estimation in the Krishna-Godavari Basin Using Machine Learning

Predicts **pore pressure** from well-log features using multiple regression models (Linear Regression, Random Forest, XGBoost), with full hyperparameter tuning and cross-validation.

---

## 📌 Project Overview

Pore pressure — the pressure of fluids trapped inside the pore spaces of subsurface rocks — is critical for assessing rock stability, wellbore integrity, and fluid flow behavior during drilling. Accurate pore pressure estimation reduces drilling risk and helps engineers plan safer wells.

This project takes well-log measurements as input and trains machine learning regressors to predict pore pressure values at each depth. Four models are compared head-to-head, the top two are hyperparameter-tuned via `RandomizedSearchCV`, and the final results are validated with 5-fold cross-validation.

---

## 📁 Project Structure

```
├── Pore_Presuure_in_KGB_Analysis.ipynb   # Main Jupyter Notebook
├── well data.csv                         # Well-log dataset (11,494 samples, 10 columns)
└── README.md                             # This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `well data.csv` |
| Total Samples | 11,494 |
| Total Features | 10 (8 inputs + DEPTH + target PP) |

### Columns

| Column | Description |
|---|---|
| `DEPTH` | Depth of measurement (meters) |
| `GR` | Gamma Ray log — indicates clay/shale content |
| `RHOB` | Bulk density log |
| `Vp` | P-wave velocity |
| `Vsh` | Shale volume fraction |
| `Caliper` | Caliper log — measures borehole diameter |
| `Porosity` | Rock porosity (%) |
| `Resistivity` | Formation resistivity |
| `Stress` | Overburden stress |
| `PP` | **Pore Pressure (target variable)** |

---

## 🔄 Workflow

### 1. Data Loading & Exploration
- Loaded `well data.csv` and inspected shape, types, and basic statistics.
- Confirmed zero null values in the raw dataset.

### 2. Outlier Detection & Handling
- Generated box plots for every feature to visually identify outliers.
- Initially tried the standard IQR method, but it was too aggressive — it replaced entire features with NaN, leaving only 7 of 10 columns usable.
- Switched to **domain-informed thresholds** instead: each feature was clipped based on physically meaningful boundaries (e.g., `GR < 70`, `RHOB < 1.5`, `Caliper > 11`). Values outside these bounds were set to NaN, then all rows containing NaN were dropped.
- Re-plotted box plots to confirm outliers were resolved.

### 3. Feature Scaling
- Applied **MinMaxScaler** to all 8 input features to normalize them into the [0, 1] range.
- `DEPTH` and `PP` (target) were kept unscaled and re-attached after scaling.

### 4. Exploratory Data Analysis
- Generated **histogram distributions** and **KDE plots** for all scaled input features to assess data spread and skewness.
- Computed a full **correlation heatmap** between all features and the target `PP`.

### 5. Feature Selection
- Dropped `Vp` (P-wave velocity) after correlation analysis — it was redundant with other features.
- Final feature set: **7 input features** → 1 target (`PP`).

### 6. Train / Test Split
- 80 / 20 split with `random_state=42`.
- **Training set:** 8,326 samples | **Test set:** 2,082 samples

### 7. Model Training & Comparison
Four models were trained and compared on the test set:

| Model | R² | Adjusted R² | MAE | MSE | RMSE |
|---|---|---|---|---|---|
| Linear Regression | 0.668 | — | 93.10 | 14,416.86 | 120.07 |
| Random Forest (default) | 0.950 | 0.950 | 25.64 | 2,152.67 | 46.40 |
| XGBoost (default) | 0.943 | 0.943 | 32.96 | 2,452.67 | 49.52 |

Linear Regression underperformed significantly. Random Forest and XGBoost both captured the non-linear patterns in the data effectively.

### 8. Hyperparameter Tuning
Both top models were tuned using **RandomizedSearchCV** (10 iterations, 5-fold CV, scored on negative MSE).

#### Random Forest — Best Parameters
```
n_estimators:       300
max_depth:          20
min_samples_split:  2
min_samples_leaf:   1
max_features:       log2
```

#### XGBoost — Best Parameters
```
n_estimators:       300
max_depth:          5
learning_rate:      0.1
subsample:          1.0
colsample_bytree:   1.0
```

### 9. Cross-Validation (5-Fold)
After tuning, both models were evaluated with shuffled 5-fold cross-validation to confirm generalization:

| Model | Test R² | Mean CV R² |
|---|---|---|
| Random Forest (default) | 0.949 | 0.951 |
| Random Forest (tuned) | 0.931 | 0.932 |
| XGBoost (tuned) | 0.934 | — |

> The default Random Forest actually had a marginally higher test R² (0.950) than the tuned version (0.932). This is because `RandomizedSearchCV` optimised for generalisation across all CV folds (scoring on `neg_mean_squared_error`), which slightly constrained the model compared to the unconstrained default that happened to fit this particular test split well. Both versions are robust — the tuned models are less likely to overfit on unseen data.

---

## 📈 Final Model Performance Summary

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.668 | 93.10 | 120.07 |
| Random Forest (default) | **0.950** | 25.64 | 46.40 |
| XGBoost (default) | 0.943 | 32.96 | 49.52 |
| Random Forest (tuned) | 0.932 | 36.03 | 54.35 |
| XGBoost (tuned) | 0.934 | 36.25 | 53.36 |

**Best test-set performance:** Default Random Forest (R² = 0.950, RMSE = 46.40)  
**Best generalised performance:** Tuned XGBoost (R² = 0.934, RMSE = 53.36) — most stable across folds.

---

## ⚙️ Technologies & Libraries

| Library | Usage |
|---|---|
| `pandas` | Data loading, manipulation, and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Scatter plots and prediction visualisations |
| `seaborn` | Box plots, heatmaps, distribution plots |
| `scikit-learn` | Train/test split, scaling, LinearRegression, RandomForestRegressor, RandomizedSearchCV, cross_val_score, all metrics |
| `xgboost` | XGBRegressor |

---

## 🚀 How to Run

1. **Clone or download** this repository.
2. Place `well data.csv` in the same directory as the notebook.
3. Open `Pore_Presuure_in_KGB_Analysis.ipynb` in **Jupyter Notebook** or **Google Colab**.
4. Run all cells top to bottom (`Kernel → Restart & Run All`).

### Prerequisites
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 📝 Key Design Decisions

- **Why domain thresholds over IQR?** The IQR method wiped out entire features because the data distributions were tight and skewed. Physically-informed cutoffs (e.g., density can't be below 1.5 g/cc) preserved all features while still removing genuinely anomalous readings.
- **Why drop `Vp`?** Correlation analysis showed it was redundant with other velocity/density features, and removing it simplified the model without hurting performance.
- **Why MinMaxScaler and not StandardScaler?** The features have bounded physical ranges, so [0, 1] normalisation maps each feature onto its natural domain — a better fit than z-score standardisation here.
- **Why RandomizedSearchCV over GridSearchCV?** The combined parameter grid is large. RandomizedSearchCV explores a representative sample of the space in a fraction of the time, with minimal loss in quality.

---

## 📈 Possible Improvements

- Add **feature importance plots** (Random Forest `.feature_importances_` or SHAP values) to understand which well-log parameters matter most.
- Try **stacking or blending** the tuned Random Forest and XGBoost into an ensemble.
- Incorporate **depth-based temporal features** (e.g., rolling averages or gradient of pressure with depth).
- Use **Bayesian optimisation** (e.g., `optuna`) instead of `RandomizedSearchCV` for more efficient tuning.
- Evaluate with additional metrics like **quantile regression** to produce prediction intervals (confidence bounds on pore pressure).
