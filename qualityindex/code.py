from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, mean_squared_log_error, explained_variance_score
)
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

def create_model(reg_type):
    """Creates and returns the appropriate regression model based on the given type."""
    if reg_type == "linear":
        return LinearRegression()
    elif reg_type == "ridge":
        return Ridge()
    elif reg_type == "lasso":
        return Lasso()
    elif reg_type == "elasticnet":
        return ElasticNet()
    elif reg_type == "sgd":
        return SGDRegressor()
    elif reg_type == "bayesian":
        return BayesianRidge()
    elif reg_type == "decision_tree":
        return DecisionTreeRegressor()
    elif reg_type == "random_forest":
        return RandomForestRegressor()
    elif reg_type == "gradient_boosting":
        return GradientBoostingRegressor()
    elif reg_type == "xgboost":
        return XGBRegressor()
    elif reg_type == "svr":
        return SVR()
    elif reg_type == "knn":
        return KNeighborsRegressor()
    elif reg_type == "pls":
        return PLSRegression()
    elif reg_type == "pcr":
        return PCA(n_components=2)  # Principal Component Regression (PCR)
    else:
        return None

def compute_regression_metrics(X_train, X_test, y_train, y_test, delta=1.0):
    regression_types = [
        "linear", "ridge", "lasso", "elasticnet", "sgd", "bayesian", 
        "decision_tree", "random_forest", "gradient_boosting", "xgboost", 
        "svr", "knn", "pls", "pcr"
    ]
    
    results = []
    
    for reg_type in regression_types:
        model = create_model(reg_type)
        if model is None:
            continue
        
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        metrics = {
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "R² Score": r2_score(y_test, y_test_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_test_pred),
            "MSLE": mean_squared_log_error(y_test, y_test_pred) if np.all(y_test_pred > 0) and np.all(y_test > 0) else None,
            "Explained Variance": explained_variance_score(y_test, y_test_pred),
            "MBD": np.mean(y_test_pred - y_test),
            "Huber Loss": np.mean(
                np.where(
                    np.abs(y_test - y_test_pred) < delta,
                    0.5 * (y_test - y_test_pred) ** 2,
                    delta * (np.abs(y_test - y_test_pred) - 0.5 * delta)
                )
            )
        }
        
        results.append((reg_type, metrics))
    
    return results

# Example usage:
# results = compute_regression_metrics(X_poly_train, X_poly_test, y_train, y_test)
# for reg, metrics in results:
#     print(f"Regression Type: {reg}")
#     for metric, value in metrics.items():
#         print(f"  {metric}: {value}")
#     print("-" * 40)