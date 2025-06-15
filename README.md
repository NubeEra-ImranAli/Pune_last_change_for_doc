# Deep Learning Model – v1

This branch contains a deep learning (TensorFlow/Keras) and XGB model built for predicting CO2_MIN from various air quality metrics. This version achieved:

- ✅ **Accuracy**: ~96% (R² Score)
- 🧠 **Architecture**: 3 Dense layers + Dropout + BatchNormalization
- 📈 **Preprocessing**:
  - Polynomial Features (Degree 2)
  - Location-wise data splits
- 💡 Loss Function: MSE
- 🛠️ Optimizer: Adam (lr=0.001)
- ⏱️ Training: 300 epochs, early stopping, learning rate scheduler
