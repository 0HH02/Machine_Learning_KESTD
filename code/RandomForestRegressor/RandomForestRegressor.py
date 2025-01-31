import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np


# 1. Load the data
file_path = "../../DB/output.csv"
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
    "regressor__min_samples_leaf": [5],  # Un solo valor
}


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=None,
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
    joblib.dump(best_model, "random_forest_pipeline1.pkl")
