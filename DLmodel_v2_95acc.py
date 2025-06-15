import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Step 1: Load Data ===
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

for name in data['NAME'].unique():
    subset = data[data['NAME'] == name]

    X = subset[features]
    y = subset[target]

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model Architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5)
    ]

    # Training
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluation
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    total_mse += mse
    total_r2 += r2
    count += 1

    print(f"[{name}] R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")

# === Summary ===
avg_rmse = np.sqrt(total_mse / count)
avg_r2 = total_r2 / count

print("\n📊 Final Deep Learning Results with Preprocessing:")
print(f"→ Average RMSE: {avg_rmse:.4f}")
print(f"→ Average R² Score: {avg_r2:.4f}")
