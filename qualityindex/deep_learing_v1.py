import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def deep_learning(data):

    # === Feature Engineering ===
    data['CO_NO2_ratio'] = data['CO_MAX'] / (data['NO2_MAX'] + 1e-5)
    data['PM_DIFF'] = data['PM10_MAX'] - data['PM2_MAX']
    data['OZONE_NO2_ratio'] = data['OZONE_MAX'] / (data['NO2_MAX'] + 1e-5)

    if 'LASTUPDATEDATETIME' in data.columns:
        data['LASTUPDATEDATETIME'] = pd.to_datetime(data['LASTUPDATEDATETIME'])
        data['hour'] = data['LASTUPDATEDATETIME'].dt.hour
        data['day'] = data['LASTUPDATEDATETIME'].dt.day
        data['month'] = data['LASTUPDATEDATETIME'].dt.month

    # === Setup ===
    target = 'CO2_MIN'
    all_features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX',
                    'CO_NO2_ratio', 'PM_DIFF', 'OZONE_NO2_ratio', 'hour', 'day', 'month']

    total_mse, total_r2, total_mae, count = 0, 0, 0, 0
    start_time = time.time()

    for name in data['NAME'].unique():
        subset = data[data['NAME'] == name].copy()
        X = subset[all_features]
        y = subset[target]

        if y.isnull().any():
            y.fillna(y.mode()[0], inplace=True)

        X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
            X, y, test_size=0.2, random_state=42)

        for col in X_train_shap.columns:
            if X_train_shap[col].isnull().any():
                X_train_shap.loc[:, col].fillna(X_train_shap[col].mode()[0], inplace=True)

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
        xgb_model.fit(X_train_shap, y_train_shap)

        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test_shap)

        shap.plots.bar(shap_values, show=False)
        plt.title(f'SHAP Feature Importance - {name}')
        plt.tight_layout()
        plt.savefig(f'shap_{name}.png')
        plt.close()

        shap_importance = np.abs(shap_values.values).mean(axis=0)
        important_indices = np.argsort(shap_importance)[::-1][:8]  # Using top 8 features
        important_features = [X.columns[i] for i in important_indices]

        X_important = X[important_features].copy()
        for col in X_important.columns:
            if X_important[col].isnull().any():
                X_important.loc[:, col].fillna(X_important[col].mode()[0], inplace=True)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_important)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        # === Deep Learning ===
        dl_model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1)
        ])

        dl_model.compile(optimizer=Adam(0.0005), loss='huber_loss')

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

        # === Re-train XGBoost ===
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        # === Gradient Boosting Meta Model ===
        meta_X = np.vstack([dl_preds, xgb_preds]).T
        meta_model = GradientBoostingRegressor()
        meta_model.fit(meta_X, y_test)
        ensemble_preds = meta_model.predict(meta_X)

        mse = mean_squared_error(y_test, ensemble_preds)
        r2 = r2_score(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)

        print(f"{name} - RMSE: {np.sqrt(mse):.3f}, R²: {r2:.4f}, MAE: {mae:.3f}")
        total_mse += mse
        total_r2 += r2
        total_mae += mae
        count += 1

    end_time = time.time()
    rmse_total = np.sqrt(total_mse / count)
    r2_avg = total_r2 / count
    mae_avg = total_mae / count

    print(f"\n⏱ Time: {end_time - start_time:.2f}s")
    print(f"📊 Avg RMSE: {rmse_total:.3f}")
    print(f"📈 Avg R²: {r2_avg:.4f}")
    print(f"📉 Avg MAE: {mae_avg:.3f}")
    return f"{rmse_total:.3f}", f"{r2_avg:.4f}", f"{mae_avg:.3f}", f"{end_time - start_time:.2f}"