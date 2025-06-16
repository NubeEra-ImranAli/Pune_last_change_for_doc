
# Ensemble: Deep Learning + XGBoost + Optuna hyperparameter tuning

This branch contains an ensemble model that averages predictions from a trained deep learning model, SHAP Feature Importance and XGBoost regressor.

- ✅ **Accuracy**: ~99% (R² Score)
- ⚡️ **Ensemble Method**: Soft Average
- 🤖 Models:
  - Deep Neural Network (Keras)
  - XGBoost Regressor
  - Optuna hyperparameter tuning
- 📈 **Preprocessing**:
  - Polynomial Features (Degree 2)
  - StandardScaler
- 🧪 Location-based test split for robustness

## File
- `XGB-Hyper-optuna.py`: Contains model definitions, ensemble logic, and performance metrics.

## Run
```bash
python XGB-Hyper-optuna.py
