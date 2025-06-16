
# Ensemble: Deep Learning + XGBoost + SHAP Feature Importance

This branch contains an ensemble model that averages predictions from a trained deep learning model, SHAP Feature Importance and XGBoost regressor.

- ✅ **Accuracy**: ~99% (R² Score)
- ⚡️ **Ensemble Method**: Soft Average
- 🤖 Models:
  - Deep Neural Network (Keras)
  - XGBoost Regressor
  - SHAP feature importance
- 📈 **Preprocessing**:
  - Polynomial Features (Degree 2)
  - StandardScaler
- 🧪 Location-based test split for robustness
  
- ✅ Final Model Comparison Table
- Model	                        MAE	      RMSE	    R² Score	    Remarks
- SHAP Feature Importance (XGB)	  0.968	    7.729	      0.9928	   ⚡ Best RMSE overall
- DL + XGB Stacking	              1.006	    7.970	      0.9929	   ⚡ Best R², strong ensemble
- Tuned XGBoost                  	0.944	    13.040	    0.9928	      Great MAE, slightly higher RMSE

## File
- `shapModel.py`: Contains model definitions, ensemble logic, and performance metrics.

## Run
```bash
python shapModel.py
