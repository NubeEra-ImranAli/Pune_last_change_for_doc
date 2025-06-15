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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
data = pd.read_csv("/content/Dataset.csv")

# === Step 2: IQR Outlier Removal ===
exclude_cols = ['NAME']
numeric_cols = data.select_dtypes(include=np.number).columns.difference(exclude_cols)


# === Step 3: Fill Missing Values Using Mode ===
for col in data.columns:
    if col not in exclude_cols and data[col].isnull().any():
        data[col].fillna(data[col].mode()[0], inplace=True)

# === Step 4: Feature/Target Setup ===
target = 'CO2_MIN'
features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

# === Step 5: Deep Learning Training per Location ===
total_mse = 0
total_r2 = 0
count = 0
start_time = time.time()
for name in data['NAME'].unique():
    subset = data[data['NAME'] == name]
    X = subset[features]
    y = subset[target]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Deep Learning Model
    dl_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    dl_model.compile(optimizer=Adam(0.001), loss='mse')
    dl_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5)
        ]
    )
    dl_preds = dl_model.predict(X_test).flatten()

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # === Meta-model (Stacking) ===
    meta_X = np.vstack([dl_preds, xgb_preds]).T
    meta_model = Ridge()
    meta_model.fit(meta_X, y_test)
    ensemble_preds = meta_model.predict(meta_X)

    # === Metrics ===
    mse = mean_squared_error(y_test, ensemble_preds)
    r2 = r2_score(y_test, ensemble_preds)

    print(f"{name} - RMSE: {np.sqrt(mse):.3f}, R²: {r2:.4f}")

    total_mse += mse
    total_r2 += r2
    count += 1

end_time = time.time()
rmse_total = np.sqrt(total_mse / count)
r2_avg = total_r2 / count

print(f"\n⏱ Total time: {end_time - start_time:.2f}s")
print(f"📊 Average RMSE: {rmse_total:.3f}")
print(f"📈 Average R²: {r2_avg:.4f}")
