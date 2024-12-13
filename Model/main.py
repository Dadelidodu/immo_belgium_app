import os
import pandas as pd
import seaborn as sns
from Scripts.Clean import clean_dataset
from Scripts.Preprocess import preprocess_dataset
from Scripts.Model import build_model, train_model, shap_analysis
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import time


import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file into a DataFrame
script_dir = os.path.dirname(os.path.abspath(__file__))
scraping_dataset_path = os.path.join(script_dir, 'data/scraping_data.csv')
df = pd.read_csv(scraping_dataset_path)

# Cleaning dataset
df = clean_dataset(df)

# Preprocess cleaned dataset
X_train, X_test, y_train, y_test = preprocess_dataset(df)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Build the model
model = build_model(X_train)

# Train the model
start_time = time.time()
trained_model = train_model(model, X_train, y_train, X_test, y_test)
end_time = time.time()
training_time = end_time - start_time
model.save("data/trained_model.h5")

# Evaluate the model
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

# R² score
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# MAPE (Mean Absolute Percentage Error) with zero check
def calculate_mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

train_mape = calculate_mape(y_train, y_train_pred)
test_mape = calculate_mape(y_test, y_test_pred)

# sMAPE (Symmetric Mean Absolute Percentage Error) with zero check
def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator[denominator == 0] = 1  # Avoid division by zero
    return np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100

train_smape = calculate_smape(y_train, y_train_pred)
test_smape = calculate_smape(y_test, y_test_pred)

# Time inference for single prediction


start_inference_time = time.time()
X_test_array = X_test.to_numpy()
single_sample = X_test_array[0].reshape(1, -1)
single_prediction = model.predict(single_sample)
end_inference_time = time.time()

inference_time = end_inference_time - start_inference_time


# Print results
print("Training MAE (from model):", train_mae)
print("Test MAE (from model):", test_mae,'\n')
print("Training R²:", train_r2)
print("Test R²:", test_r2,'\n')
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse,'\n')
print("Training MAPE:", train_mape)
print("Test MAPE:", test_mape,'\n')
print("Training sMAPE:", train_smape)
print("Test sMAPE:", test_smape,'\n')
print(f"Training Time: {training_time:.2f} seconds")
print(f"Single Prediction Time: {inference_time:.2f} seconds",'\n')

# SHAP Analysis

background, X_test_sample, explainer, shap_values = shap_analysis(X_train, X_test, model)

# Plot SHAP values results

shap_values_reshaped = np.squeeze(shap_values)
shap_df = pd.DataFrame(shap_values_reshaped, columns=X_test_sample.columns)
shap_df_abs = shap_df.abs().mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.heatmap(shap_df_abs.to_frame().T, annot=True, cmap="coolwarm")
plt.title("Feature Importance Heatmap")
plt.tight_layout()
plt.show()

# Plot training loss results

plt.figure(figsize=(12, 6))
plt.plot(trained_model.history['loss'][15:], label='Training Loss', color='blue', linestyle='-', linewidth=2)
plt.plot(trained_model.history['val_loss'][15:], label='Test Loss', color='orange', linestyle='--', linewidth=2)
plt.scatter(range(len(trained_model.history['loss'][15:])), trained_model.history['loss'][15:], color='blue', s=10, alpha=0.6)
plt.scatter(range(len(trained_model.history['val_loss'][15:])), trained_model.history['val_loss'][15:], color='orange', s=10, alpha=0.6)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel('Epochs (Starting from 10)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
plt.title('Training vs. Validation Loss Over Epochs', fontsize=16, fontweight='bold')

plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, borderpad=1, fancybox=True)

plt.fill_between(range(len(trained_model.history['loss'][15:])), 
                 trained_model.history['loss'][15:], 
                 trained_model.history['val_loss'][15:], 
               trained_model.history['val_loss'][15:], 
                 color='gray', alpha=0.2, label='Loss Difference')

min_val_loss_epoch = np.argmin(trained_model.history['val_loss'])
plt.annotate(f'Min Val Loss: {trained_model.history["val_loss"][min_val_loss_epoch]:.4f}',
             xy=(min_val_loss_epoch, trained_model.history['val_loss'][min_val_loss_epoch]),
             xytext=(min_val_loss_epoch, trained_model.history['val_loss'][min_val_loss_epoch] + 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, color='darkred')

plt.tight_layout()
plt.show()
