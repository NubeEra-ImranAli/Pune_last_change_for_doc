
import pandas as pd

from sklearn.decomposition import PCA

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,SGDRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

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


def compute_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

split_data = {}

for name in data['NAME'].unique():
    subset_data = data[data['NAME'] == name]

    X = subset_data[features]
    y = subset_data[target]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Store the split data in the dictionary
    split_data[name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
def perform_regression(split_data, regression_type="linear"):
    start_time = time.time()

    mse_list = []
    r2_list = []
    
    for name in split_data:
        X_train = split_data[name]['X_train']
        X_test = split_data[name]['X_test']
        y_train = split_data[name]['y_train']
        y_test = split_data[name]['y_test']
        
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
            mse, r2 = compute_metrics(model, X_train_pca, X_test_pca, y_train, y_test)
            mse_list.append(mse)
            r2_list.append(r2)
            continue
        else:
            raise ValueError(f"Unsupported regression type: {regression_type}")
        
        mse, r2 = compute_metrics(model, X_poly_train, X_poly_test, y_train, y_test)
        mse_list.append(mse)
        r2_list.append(r2)
    
    total_test_mse = sum(mse_list)
    avg_r2_score = sum(r2_list) / len(r2_list)
    
    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = "{:.2f}".format(execution_time)
    
    print(f"Root Mean Squared Error (RMSE) for {regression_type} - Test: {total_test_mse}")
    print(f"Average R² Score for {regression_type}: {avg_r2_score}")
    print(f"Execution Time for {regression_type}: {formatted_time} seconds")
    
    return total_test_mse, avg_r2_score, formatted_time

# List of regression types
# regression_types = [
#     "linear", "ridge", "lasso", "elasticnet", "sgd", 
#     "bayesian", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "svr", "knn", 
#     "pls", "pcr"
# ]

regression_types = [
    "linear", "ridge", "sgd", 
    "bayesian", "decision_tree", "xgboost", "knn", 
    "pls", "pcr"
]

# Dictionary to store results
all_results = {}

# Loop through all regression types
for reg_type in regression_types:
    print(f"\nPerforming regression: {reg_type}")
    
    # Perform the regression and get the metrics
    total_test_mse, avg_r2_score, exec_time = perform_regression(split_data, regression_type=reg_type)
    
    # Store the results for each regression type
    all_results[reg_type] = {
        'total_test_mse': total_test_mse,
        'avg_r2_score': avg_r2_score,
        'execution_time': exec_time
    }

# Example: print results for linear regression
print("\nLinear Regression Results:")
print(f"Total RMSE: {all_results['linear']['total_test_mse']}")
print(f"Average R² Score: {all_results['linear']['avg_r2_score']}")
print(f"Execution Time: {all_results['linear']['execution_time']} seconds")

# Print results for all regression types
for reg_type in regression_types:
    print(f"\n{reg_type.capitalize()} Regression Results:")
    print(f"Total RMSE: {all_results[reg_type]['total_test_mse']}")
    print(f"Average R² Score: {all_results[reg_type]['avg_r2_score']}")
    print(f"Execution Time: {all_results[reg_type]['execution_time']} seconds")