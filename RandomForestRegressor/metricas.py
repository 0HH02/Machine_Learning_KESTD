import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Cargar el modelo entrenado
model = joblib.load("RandomForestRegressor/random_forest_pipeline.pkl")

# 2. Cargar el conjunto de prueba (X_test) y la variable objetivo real (y_test)
file_path = "DB/output.csv"  # Asegúrate de usar el archivo correcto
data = pd.read_csv(file_path)

# 3. Preparar los datos (ajustar según los mismos pasos de preprocesamiento usados antes)

# 4. Separate features and target
X = data.drop(columns=["007_score_ligando"])
y = data["007_score_ligando"]

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Predecir valores con el modelo cargado
y_pred = model.predict(X_test)

# Plot real vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, s=10, edgecolor="k", label="Data Points")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    "r--",
    label="Perfect Fit Line",
)
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Ensure y_test and y_pred are numpy arrays for proper indexing
y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

# Sort values by actual values for smoother visualization
sorted_indices = np.argsort(y_test_array)
y_test_sorted = y_test_array[sorted_indices]
y_pred_sorted = y_pred_array[sorted_indices]

# Plot sorted actual and predicted values using smaller points
plt.figure(figsize=(10, 6))

# Plot actual values as small points
plt.scatter(
    range(len(y_test_sorted)),
    y_test_sorted,
    label="Actual Values (y_test)",
    marker=".",
    s=10,
    alpha=0.7,
)

# Plot predicted values as small points
plt.scatter(
    range(len(y_pred_sorted)),
    y_pred_sorted,
    label="Predicted Values (y_pred)",
    marker=".",
    s=10,
    alpha=0.7,
)

# Adding labels, legend, and title
plt.xlabel("Sorted Index")
plt.ylabel("Values")
plt.title("Actual vs Predicted Values (Sorted, Small Points)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Suponiendo que ya tienes y_test y y_pred con los valores reales y predichos

# Calcular métricas de error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Raíz del error cuadrático medio
mae = mean_absolute_error(y_test, y_pred)  # Error absoluto medio
r2 = r2_score(y_test, y_pred)  # Coeficiente de determinación R²

# Imprimir estadísticas
print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")
