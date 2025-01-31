import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# automl = autosklearn.regression.AutoSklearnRegressor(
#     time_left_for_this_task=1800,
#     per_run_time_limit=90,
#     ensemble_kwargs={'ensemble_size': 100},
#     max_models_on_disc=50,  
#     seed=42
# )

model_path = "third_automl_model.pkl"
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


# Mean Squared Error (MSE): 0.0133
# R^2 Score: 0.7939
# Mean Absolute Error (MAE): 0.0893

# Ranking de los modelos usados:
#           rank  ensemble_weight               type      cost   duration
# model_id                                                               
# 59           1             0.14  gradient_boosting  0.210635  21.876693
# 55           2             0.24  gradient_boosting  0.213652  19.059410
# 70           3             0.26  gradient_boosting  0.213792  21.179785
# 65           4             0.14  gradient_boosting  0.215000  20.202920
# 72           5             0.08  gradient_boosting  0.217026  34.821860
# 29           6             0.08  gradient_boosting  0.219838  77.496639
# 33           7             0.06  gradient_boosting  0.220113  11.866208

# Mejor modelo:
# {29: {'model_id': 29, 'rank': 1, 'cost': 0.2198382229860153, 'ensemble_weight': 0.08, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67de3eca10>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67de3e0ed0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67de6f94d0>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=0.004876587483107972,
#                               learning_rate=0.01163114896746256, max_iter=512,
#                               max_leaf_nodes=127, min_samples_leaf=4,
#                               n_iter_no_change=19, random_state=42,
#                               validation_fraction=None, warm_start=True)}, 33: {'model_id': 33, 'rank': 2, 'cost': 0.22011292507048097, 'ensemble_weight': 0.06, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67dea7e2d0>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67de7c0c90>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67de6f91d0>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=0.00015248187303095465,
#                               learning_rate=0.04322652081390701, max_iter=512,
#                               max_leaf_nodes=17, min_samples_leaf=82,
#                               n_iter_no_change=19, random_state=42,
#                               validation_fraction=None, warm_start=True)}, 55: {'model_id': 55, 'rank': 3, 'cost': 0.21365225625672157, 'ensemble_weight': 0.24, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67de3ac690>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67de328590>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67de328510>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=2.115540904393944e-09,
#                               learning_rate=0.08901955203445579, max_iter=512,
#                               max_leaf_nodes=45, min_samples_leaf=31,
#                               n_iter_no_change=0, random_state=42,
#                               validation_fraction=None, warm_start=True)}, 59: {'model_id': 59, 'rank': 4, 'cost': 0.21063498786070278, 'ensemble_weight': 0.14, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67ddfe4ed0>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67de6b5cd0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67de328550>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=2.115540904393944e-09,
#                               learning_rate=0.047141532607386634, max_iter=512,
#                               max_leaf_nodes=40, n_iter_no_change=0,
#                               random_state=42, validation_fraction=None,
#                               warm_start=True)}, 65: {'model_id': 65, 'rank': 5, 'cost': 0.21499984625844493, 'ensemble_weight': 0.14, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67de359450>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67de35be50>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67ddf93950>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=4.012767470167921e-10,
#                               learning_rate=0.10000000000000002, max_iter=512,
#                               min_samples_leaf=23, n_iter_no_change=0,
#                               random_state=42, validation_fraction=None,
#                               warm_start=True)}, 70: {'model_id': 70, 'rank': 6, 'cost': 0.21379167621222983, 'ensemble_weight': 0.26, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67dd7fb550>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67dd788f10>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67dd78b450>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=0.0662552337476642,
#                               learning_rate=0.04975827738160349, max_iter=512,
#                               max_leaf_nodes=382, min_samples_leaf=137,
#                               n_iter_no_change=1, random_state=42,
#                               validation_fraction=None, warm_start=True)}, 72: {'model_id': 72, 'rank': 7, 'cost': 0.2170261084853763, 'ensemble_weight': 0.08, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f67dd78d350>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f67dcfcbdd0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f67dd78b410>, 'sklearn_regressor': HistGradientBoostingRegressor(l2_regularization=2.2543199070269395e-09,
#                               learning_rate=0.08901955203445579, max_iter=512,
#                               max_leaf_nodes=80, min_samples_leaf=28,
#                               n_iter_no_change=0, random_state=42,
#                               validation_fraction=None, warm_start=True)}}
