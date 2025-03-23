
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import folium
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
# loading dataset of smartcity Pune
df = pd.read_csv(r"D:\\nubeera\\Pune\\Dataset.csv")


data = df
# Example: if the AQI is represented as 'AirQualityIndex' in your data
# Adjust the target variable and features accordingly
target = 'CO2_MIN'
features = ['PM10_MAX', 'PM2_MAX', 'NO2_MAX', 'CO_MAX', 'OZONE_MAX']

from sklearn.preprocessing import PolynomialFeatures
import time
start_time = time.time()
# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame()

for name in data['NAME'].unique():
    subset_data = data[data['NAME'] == name]

    X = subset_data[features]
    y = subset_data[target]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    polynomial_features = PolynomialFeatures(degree=2)
    X_poly_train = polynomial_features.fit_transform(X_train)
    X_poly_test = polynomial_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    subset_predictions_test = model.predict(X_poly_test)

    test_mse = mean_squared_error(y_test, subset_predictions_test)

    subset_results = pd.DataFrame({
        'NAME': [name] * len(subset_predictions_test),
        'Predicted_AQI': subset_predictions_test,
        'Actual_AQI': y_test
    })
    predictions_df = pd.concat([predictions_df, subset_results], ignore_index=True)

# print(predictions_df)
# End the timer
end_time = time.time()
# Calculate and print the execution time
execution_time = end_time - start_time
formatted_time = "{:.2f}".format(execution_time)
print(f"Execution Time: for LinearRegression OG {formatted_time} seconds")

def perform_regression(data, features, target, regression_type="linear"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,SGDRegressor)
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    import xgboost as xgb
    from sklearn.neighbors import KNeighborsRegressor
    start_time = time.time()
    
    predictions_df = pd.DataFrame()

    for name in data['NAME'].unique():
        subset_data = data[data['NAME'] == name]

        X = subset_data[features]
        y = subset_data[target]

        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        polynomial_features = PolynomialFeatures(degree=2)
        X_poly_train = polynomial_features.fit_transform(X_train)
        X_poly_test = polynomial_features.fit_transform(X_test)

        if regression_type == "linear":
            model = LinearRegression()
        elif regression_type == "ridge":
            model = Ridge(solver='svd')
        elif regression_type == "lasso":
            model = Lasso(alpha=0.1)
        elif regression_type == "elasticnet":
            model = ElasticNet(alpha=0.1)
        elif regression_type == "sgd":
            model = SGDRegressor(max_iter=1000, tol=1e-3)
        elif regression_type == "bayesian":
            model = BayesianRidge()
        elif regression_type == "decision_tree":
            model = DecisionTreeRegressor()
        elif regression_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100)
        elif regression_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100)
        elif regression_type == "xgboost":
            model = xgb.XGBRegressor(objective='reg:squarederror')
        elif regression_type == "svr":
            model = SVR(kernel='rbf')
        elif regression_type == "knn":
            model = KNeighborsRegressor(n_neighbors=5)
        elif regression_type == "pls":
            model = PLSRegression(n_components=3)
        elif regression_type == "pcr":
            pca = PCA(n_components=3)
            X_train_pca = pca.fit_transform(X_poly_train)
            X_test_pca = pca.transform(X_poly_test)
            model = LinearRegression()
            model.fit(X_train_pca, y_train)
            subset_predictions_test = model.predict(X_test_pca)
        else:
            raise ValueError(f"Unsupported regression type: {regression_type}")

        if regression_type != "pcr":
            model.fit(X_poly_train, y_train)
            subset_predictions_test = model.predict(X_poly_test)

        # test_mse = mean_squared_error(y_test, subset_predictions_test)
        # print(f"Mean Squared Error for {name} - Test: {test_mse}")

        subset_results = pd.DataFrame({
            'NAME': [name] * len(subset_predictions_test),
            'Predicted_AQI': subset_predictions_test,
            'Actual_AQI': y_test
        })
        predictions_df = pd.concat([predictions_df, subset_results], ignore_index=True)

    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = "{:.2f}".format(execution_time)
    print(f"Execution Time for {regression_type}: {formatted_time} seconds")

    return predictions_df, formatted_time

# List of regression types
regression_types = [
    "linear", "ridge", "lasso", "elasticnet", "sgd", 
    "bayesian","decision_tree", "random_forest", "gradient_boosting", "xgboost", "svr", "knn",
    "pls","pcr"
]

# Dictionary to store results
all_results = {}

# Loop through all regression types
for reg_type in regression_types:
    print(f"\nPerforming regression: {reg_type}")
    predictions, exec_time = perform_regression(data, features, target, regression_type=reg_type)
    all_results[reg_type] = {
        'predictions': predictions,
        'execution_time': exec_time
    }

# Now you can access the predictions and execution time for each regression type
# Example: print results for linear regression
print("\nLinear Regression Results:")
print(all_results['linear']['predictions'])
print(f"Execution Time: {all_results['linear']['execution_time']} seconds")
"""#                                                END"""