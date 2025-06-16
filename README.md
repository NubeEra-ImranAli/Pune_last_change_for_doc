
# Ensemble: Deep Learning + XGBoost

This branch contains an ensemble model that averages predictions from a trained deep learning model, SHAP Feature Importance and XGBoost regressor.

- ✅ **Accuracy**: ~99% (R² Score)
- ⚡️ **Ensemble Method**: Soft Average
- 🤖 Models:
  - Deep Neural Network (Keras)
  - XGBoost Regressor
- 📈 **Preprocessing**:
  - Polynomial Features (Degree 2)
  - StandardScaler
- 🧪 Location-based test split for robustness

## File
- `EnsembleModel.py`: Contains model definitions, ensemble logic, and performance metrics.

## Run
```bash
python EnsembleModel.py
