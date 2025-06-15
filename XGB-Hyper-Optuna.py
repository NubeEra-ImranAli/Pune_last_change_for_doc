# ✅ Full Script: XGBoost Hyperparameter Tuning using Optuna

import pandas as pd
import numpy as np
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

# === Load and Prepare Data ===
data = pd.read_csv("/content/Dataset.csv")

# Feature Engineering (optional)
data['CO_NO2_ratio'] = data['CO_MAX'] / (data['NO2_MAX'] + 1e-5)
data['PM_DIFF'] = data['PM10_MAX'] - data['PM2_MAX']

features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX', 'CO_NO2_ratio', 'PM_DIFF']
target = 'CO2_MIN'

# Tuning for one location for demonstration
name = data['NAME'].unique()[0]
subset = data[data['NAME'] == name]

X = subset[features]
y = subset[target]

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Optuna Tuning ===
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "verbosity": 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# === Final Model with Best Params ===
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("Best RMSE:", rmse)
print("R² Score:", r2)
print("Best Parameters:", best_params)
