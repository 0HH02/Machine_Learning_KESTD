import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# threshold = 0.8
# cols_to_keep = data.columns[data.isnull().mean() < threshold]
# data = data[cols_to_keep]

# automl = autosklearn.regression.AutoSklearnRegressor(
#     time_left_for_this_task=300,  
#     per_run_time_limit=30,      
#     ensemble_kwargs = {'ensemble_size': 50},  
#     seed=42                     
# )
# 

model_path = "fifth_automl_model.pkl"
automl = joblib.load(model_path)

file_path = "DB/output_for_automl.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['007_score_ligando'])
y = data['007_score_ligando']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = automl.predict(X_test)

# Métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nRanking de los modelos usados:")
print(automl.leaderboard())

print("\nMejor modelo:")
print(automl.show_models())

print("\nEstadísticas del entrenamiento:")
print(automl.sprint_statistics())

print("\nOtras métricas del mejor modelo:")
models_with_weights = automl.get_models_with_weights()
for weight, model in models_with_weights:
    print(f"Weight: {weight}, Model: {model}")

# Gráfico 1: Dispersión de valores reales vs predichos
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed', color='red')  # Línea ideal
plt.xlabel("Valores Reales (y_test)")
plt.ylabel("Valores Predichos (y_pred)")
plt.title("Comparación de Valores Reales vs Predichos")
plt.show()

# Gráfico 2: Histograma de errores
plt.figure(figsize=(6, 4))
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.axvline(x=0, color='red', linestyle='dashed')
plt.xlabel("Error (y_test - y_pred)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores del Modelo")
plt.show()


y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

sorted_indexes = np.argsort(y_test_array)
y_test_sorted = y_test_array[sorted_indexes]
y_pred_sorted = y_pred_array[sorted_indexes]

# Gráfico 3: Comparación de valores reales y predichos ordenados
plt.figure(figsize=(8, 4))
plt.scatter(range(len(y_test_sorted)), y_test_sorted, label="Valores reales", s=10, alpha=0.6)
plt.scatter(range(len(y_pred_sorted)), y_pred_sorted, label="Valores predichos", s=10, alpha=0.6)
plt.xlabel("Índices ordenados")
plt.ylabel("Valor")
plt.title("Comparación de la Tendencia de Valores Reales y Predichos")
plt.legend()
plt.show()


# Mean Squared Error (MSE): 0.0139
# R^2 Score: 0.7841
# Mean Absolute Error (MAE): 0.0911

# Ranking de los modelos usados:
#           rank  ensemble_weight               type      cost   duration
# model_id                                                               
# 20           1             0.52  gradient_boosting  0.226435  14.378886
# 6            2             0.48  gradient_boosting  0.229049  22.187421

# Mejor modelo:
# {6: {'model_id': 6, 'rank': 1, 'cost': 0.2290492508328199, 'ensemble_weight': 0.48, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f9f317d7750>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f9f3168a710>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f9f31773d90>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=3.20414911914726e-10,
#                               learning_rate=0.1279702444496931, max_iter=512,
#                               max_leaf_nodes=63, min_samples_leaf=19,
#                               n_iter_no_change=8, random_state=42,
#                               validation_fraction=None, warm_start=True)}, 20: {'model_id': 20, 'rank': 2, 'cost': 0.2264353335502972, 'ensemble_weight': 0.52, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f9f316c51d0>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f9f313c8e10>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f9f313c8fd0>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=5.997418027353535e-10,
#                               learning_rate=0.12286466971783992, max_iter=512,
#                               max_leaf_nodes=26, min_samples_leaf=8,
#                               n_iter_no_change=0, random_state=42,
#                               validation_fraction=None, warm_start=True)}}