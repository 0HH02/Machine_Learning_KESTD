from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Definir el modelo base
gb_model = HistGradientBoostingRegressor(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_iter': [100, 200, 300, 400, 500],
    'max_leaf_nodes': [10, 20, 50, 100, 200, 512],
    'min_samples_leaf': [1, 5, 10, 20, 50, 100],
    'l2_regularization': np.logspace(-10, 1, 10)
}

# Configurar la búsqueda aleatoria
random_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=param_dist,
    n_iter=50,  # Número de combinaciones a probar
    cv=5,  # Cross-validation con 5 folds
    scoring='r2',
    n_jobs=-1,  # Usar todos los núcleos disponibles
    random_state=42
)


file_path = "DB/output_for_automl.csv"  
data = pd.read_csv(file_path)

X = data.drop(columns=['007_score_ligando'])
y = data['007_score_ligando']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar la búsqueda
random_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# Evaluar el mejor modelo
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResultados con mejores hiperparámetros:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
