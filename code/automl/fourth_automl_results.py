import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# automl = autosklearn.regression.AutoSklearnRegressor(
#     time_left_for_this_task=300,
#     per_run_time_limit=30,
#     ensemble_kwargs={'ensemble_size': 50},
#     resampling_strategy='cv', 
#     resampling_strategy_arguments={'folds': 5},  
#     seed=42
# )

model_path = "fourth_automl_model.pkl"
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

# Mean Squared Error (MSE): 0.0310
# R^2 Score: 0.5198
# Mean Absolute Error (MAE): 0.1411

# Ranking de los modelos usados:
#           rank  ensemble_weight           type      cost  duration
# model_id                                                          
# 14           1              1.0  liblinear_svr  0.497267  5.710828

# Mejor modelo:
# {14: {'model_id': 14, 'rank': 1, 'cost': 0.4972665093521863, 'ensemble_weight': 1.0, 'voting_model': VotingRegressor(estimators=None), 'estimators': [{'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1c61654e10>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1c612a0750>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1c63dd2ad0>, 'sklearn_regressor': LinearSVR(C=0.036388789196735646, dual=False, epsilon=0.0012739627397164333,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.020180468804448126)}, {'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1c60ecb150>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1c60efcc10>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1c60efcd50>, 'sklearn_regressor': LinearSVR(C=0.036388789196735646, dual=False, epsilon=0.0012739627397164333,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.020180468804448126)}, {'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1c60bf9e90>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1c60b4e990>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1c60b4ead0>, 'sklearn_regressor': LinearSVR(C=0.036388789196735646, dual=False, epsilon=0.0012739627397164333,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.020180468804448126)}, {'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1c6169e890>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1c61651050>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1c616516d0>, 'sklearn_regressor': LinearSVR(C=0.036388789196735646, dual=False, epsilon=0.0012739627397164333,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.020180468804448126)}, {'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1c606d92d0>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1c6069fa50>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1c6069fc90>, 'sklearn_regressor': LinearSVR(C=0.036388789196735646, dual=False, epsilon=0.0012739627397164333,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.020180468804448126)}]}}