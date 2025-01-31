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

# min_val = data.min().min() 
# placeholder_value = min_val - 9964.995
# data.fillna(placeholder_value, inplace=True) 

# # print(min_val) -35.005
# # print(max_val) 2854.0

# automl = autosklearn.regression.AutoSklearnRegressor(
#     time_left_for_this_task=300,
#     per_run_time_limit=30, 
#     ensemble_kwargs={'ensemble_size': 20},  
#     initial_configurations_via_metalearning=0,  # Desactivar bias inicial de Gradient Boosting
#     seed=42
# )

model_path = "seventh_automl_model.pkl"
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

# Mean Squared Error (MSE): 0.0209
# R^2 Score: 0.6766
# Mean Absolute Error (MAE): 0.1139

# Ranking de los modelos usados:
#           rank  ensemble_weight                 type      cost   duration
# model_id                                                                 
# 18           1             0.55  k_nearest_neighbors  0.374991  16.617969
# 13           2             0.40        decision_tree  0.418234   1.085169
# 5            3             0.05       ard_regression  0.486163  24.295314

# Mejor modelo:
# {5: {'model_id': 5, 'rank': 1, 'cost': 0.4861634721619076, 'ensemble_weight': 0.05, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f426c305f90>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f426c671ed0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f426c671f50>, 'sklearn_regressor': ARDRegression(alpha_1=4.401597425986298e-05, alpha_2=3.983035763349455e-07,
#               copy_X=False, lambda_1=4.2514887688001695e-10,
#               lambda_2=5.330858363337201e-05,
#               threshold_lambda=1194.941081664533, tol=0.004650940203487976)}, 13: {'model_id': 13, 'rank': 2, 'cost': 0.4182339116275676, 'ensemble_weight': 0.4, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f426c275a10>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f426c313290>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f426c671410>, 'sklearn_regressor': DecisionTreeRegressor(criterion='friedman_mse', max_depth=516,
#                       min_samples_leaf=13, min_samples_split=4,
#                       random_state=42)}, 18: {'model_id': 18, 'rank': 3, 'cost': 0.3749910413386486, 'ensemble_weight': 0.55, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f426c2daa10>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f426bb1d1d0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f426c3138d0>, 'sklearn_regressor': KNeighborsRegressor(n_neighbors=21, p=1, weights='distance')}}