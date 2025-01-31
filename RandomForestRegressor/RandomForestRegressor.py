import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load the data
file_path = "DB/output.csv"
data = pd.read_csv(file_path)

# 2. Drop columns with >80% null values
threshold = 0.8
cols_to_keep = data.columns[data.isnull().mean() < threshold]
data = data[cols_to_keep]

# 3. Drop rows with null target values
data = data.dropna(subset=["007_score_ligando"])

# 4. Separate features and target
X = data.drop(columns=["007_score_ligando"])
y = data["007_score_ligando"]

# Debug target
print("Target (y) null count:", y.isnull().sum())
print("Target (y) dtype:", y.dtypes)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Debug features
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 5. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

# 6. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Debug transformed X
print("Training data shape:", X_train.shape)
transformed_X = preprocessor.fit_transform(X_train)
print("Transformed X shape:", transformed_X.shape)

# 8. GridSearchCV
param_grid = {
    "regressor__n_estimators": [100],  # Un solo valor
    "regressor__max_depth": [None],  # Dos opciones para max_depth
    "regressor__min_samples_split": [2],  # Un solo valor
}


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=2,
    scoring="neg_mean_squared_error",
    error_score="raise",  # Raise errors during debugging
    n_jobs=-1,
)

# Debugging fit
try:
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
except Exception as e:
    print("Error during fitting:", str(e))
    raise

# 9. Evaluation (only if fit succeeds)
if "grid_search" in locals() and grid_search.best_estimator_:
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")
    joblib.dump(best_model, "random_forest_pipeline.pkl")


import matplotlib.pyplot as plt
import numpy as np

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


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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
