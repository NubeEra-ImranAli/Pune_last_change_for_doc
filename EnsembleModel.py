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

data = pd.read_csv("/content/Dataset.csv")

target = 'CO2_MIN'
features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

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

    # Ensemble Prediction
    ensemble_preds = (dl_preds + xgb_preds) / 2

    mse = mean_squared_error(y_test, ensemble_preds)
    r2 = r2_score(y_test, ensemble_preds)

    total_mse += mse
    total_r2 += r2
    count += 1

end_time = time.time()
rmse_total = np.sqrt(total_mse)
r2_avg = total_r2 / count

rmse_total, r2_avg
