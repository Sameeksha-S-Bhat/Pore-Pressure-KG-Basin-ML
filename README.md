# 🛢️ Pore Pressure Estimation in the Krishna-Godavari (KG) Basin Using Machine Learning

A machine learning project that predicts subsurface pore pressure from well-log sensor readings in the Krishna-Godavari (KG) Basin, India — providing a data-driven alternative to expensive direct measurement techniques.

---

## 📌 Overview

Pore pressure refers to the pressure of fluids within the pore spaces of rocks or sediments. Accurate estimation is critical for wellbore stability, drilling safety, and reservoir geomechanics. Traditional methods rely on costly physical instruments like piezometers and pressure transducers.

This project uses **well-log sensor data** from shallow offshore sediments of the KG Basin to train and compare multiple ML regression models, achieving an R² of **0.95** with Random Forest — demonstrating that sensor-based ML estimation is a viable and cost-effective approach.

---

## 📊 Dataset

- **Source:** Well-log data from the Krishna-Godavari (KG) Basin, India
- **Size:** 11,494 readings across a depth range of ~6m to ~336m
- **Target variable:** Pore Pressure (PP) in psi

### Sensor Features Used

| Feature | Description |
|---------|-------------|
| GR | Gamma Ray — measures natural radioactivity, indicates shale content |
| RHOB | Bulk Density — density of the formation (g/cc) |
| Vsh | Volume of Shale — shale fraction in the rock |
| Caliper | Measures borehole diameter (inches) |
| Porosity | Fraction of void space in the sediment (%) |
| Resistivity | Electrical resistance of the formation (ohm·m) |
| Stress | Overburden/mechanical stress at depth |

> **Note:** Vp (P-wave velocity) was excluded due to 313 rows containing the -999.25 null indicator (standard LAS format convention) and near-zero correlation (-0.008) with the target variable.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** — Data processing
- **Matplotlib, Seaborn** — Data visualization
- **Scikit-learn** — ML models, preprocessing, hypertuning
- **XGBoost** — Gradient boosting model
- **Jupyter Notebook** — Analysis environment

---

## 🔍 Methodology

### 1. Data Preprocessing
- Applied domain-specific physical thresholds for outlier removal instead of standard IQR (which eliminated entire features due to tight well-log distributions)
- Dropped rows with physically implausible sensor values based on known valid ranges
- Applied MinMaxScaler normalization to all retained features

### 2. Feature Selection
- Removed Vp due to data quality issues and near-zero correlation with target
- Retained 7 features: GR, RHOB, Vsh, Caliper, Porosity, Resistivity, Stress
- Correlation heatmap confirmed Stress, RHOB, and Resistivity as strongest predictors

### 3. Models Compared

| Model | R² Score |
|-------|----------|
| Random Forest (default) | **0.95** |
| Random Forest (tuned) | 0.93 |
| XGBoost (tuned) | ~0.93 |
| Linear Regression | 0.67 |
| SVR (RBF kernel) | 0.66 |

### 4. Hyperparameter Tuning
- Applied **RandomizedSearchCV** with 5-fold cross-validation on Random Forest and XGBoost
- Best RF parameters: `n_estimators=300, max_depth=20, max_features='log2'`
- Tuned models achieved consistent R² ≈ 0.93, confirming good generalisation without overfitting

### 5. Feature Importance

From the best Random Forest model:

| Feature | Importance |
|---------|-----------|
| Stress | 0.5453 |
| Resistivity | 0.2464 |
| RHOB | 0.1089 |
| Vsh | 0.0373 |
| Caliper | 0.0293 |
| GR | 0.0232 |
| Porosity | 0.0096 |

Stress and Resistivity dominate — consistent with geomechanical theory where overburden stress and formation fluid content are primary controls on pore pressure.

---

## 📁 Project Structure

```
├── Pore Presuure in KGB Analysis.ipynb   # Main analysis notebook
├── well data.csv                          # Well-log sensor dataset
└── README.md                             # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/Sameeksha-S-Bhat/Pore-Pressure-KG-Basin.git
```

2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

3. Launch the notebook
```bash
jupyter notebook "Pore Presuure in KGB Analysis.ipynb"
```

4. Run all cells in order

---

## 📈 Key Results

- Linear Regression (R² = 0.67) confirmed **non-linear relationships** between sensor readings and pore pressure
- Random Forest achieved **R² = 0.95** on test data, making it the best model
- SVR underperformed (R² = 0.66), showing ensemble tree-based methods are better suited for geophysical sensor data
- **Stress, Resistivity, and RHOB** were the three most important predictors — physically interpretable and consistent with domain knowledge
- Hypertuning confirmed model robustness with consistent R² ≈ 0.93 across cross-validation folds

---

## 🔑 Key Learnings

- Standard IQR outlier removal fails on well-log data due to tight feature distributions — domain-specific thresholds are more appropriate
- Tree-based ensemble models significantly outperform linear models on geophysical sensor data
- Feature importance from Random Forest aligns with geomechanical theory, validating the model's physical interpretability
- Sensor-based ML can serve as a practical, low-cost alternative to direct pore pressure measurement in data-rich basin environments

---

## 👩‍💻 Author

**Sameeksha S. Bhat**  
B.E. Artificial Intelligence and Data Science, NMIT Bangalore  
[GitHub](https://github.com/Sameeksha-S-Bhat) | [LinkedIn](https://linkedin.com/in/sameeksha-s-bhat-2a7341336)


## 📄 License
This project is licensed under the MIT License.
