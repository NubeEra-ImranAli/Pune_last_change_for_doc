from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# === Load Data ===
df = pd.read_csv("Dataset.csv")

# === Config ===
target = 'CO2_MIN'
features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

# === Clean Data (Optional: Add your outlier and NaN handling here) ===
# Here we assume data is already cleaned as you said

# === Split Data by 'NAME' ===
split_data = {}

for name in df['NAME'].unique():
    subset = df[df['NAME'] == name]
    X = subset[features]
    y = subset[target]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    split_data[name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# === Deep Learning Model Function ===
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Perform Deep Learning Regression ===
def perform_deep_learning(split_data):
    mse_list = []
    r2_list = []
    total_start = time.time()

    for name, split in split_data.items():
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = create_model(X_train_scaled.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

        preds = model.predict(X_test_scaled).flatten()
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mse_list.append(mse)
        r2_list.append(r2)

        print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")

    total_rmse = np.sqrt(sum(mse_list))
    avg_r2 = np.mean(r2_list)
    exec_time = time.time() - total_start

    print("\n🧠 Deep Learning Results:")
    print(f"→ Total RMSE: {total_rmse:.4f}")
    print(f"→ Avg R² Score: {avg_r2:.4f}")
    print(f"→ Time Taken: {exec_time:.2f} sec")

    return total_rmse, avg_r2, exec_time

# === Run the Deep Learning Model ===
rmse, r2, duration = perform_deep_learning(split_data)  # Final results
