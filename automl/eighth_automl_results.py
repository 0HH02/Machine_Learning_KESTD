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
#     include={'regressor': ['liblinear_svr']},
#     ensemble_kwargs={'ensemble_size': 50}, 
#     seed=42
# )

model_path = "eighth_automl_model.pkl"
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


# Mean Squared Error (MSE): 0.0222
# R^2 Score: 0.6558
# Mean Absolute Error (MAE): 0.1198

# Ranking de los modelos usados:
#           rank  ensemble_weight           type      cost  duration
# model_id                                                          
# 36           1             0.22  liblinear_svr  0.396142  3.258168
# 27           2             0.28  liblinear_svr  0.396579  1.729337
# 12           3             0.42  liblinear_svr  0.402007  1.264682
# 20           4             0.02  liblinear_svr  0.441820  1.296930
# 29           5             0.06  liblinear_svr  0.512327  0.958962

# Mejor modelo:
# {12: {'model_id': 12, 'rank': 1, 'cost': 0.4020066799081108, 'ensemble_weight': 0.42, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1414f29190>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f14152782d0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1414fda990>, 'sklearn_regressor': LinearSVR(C=13.973623061090379, dual=False, epsilon=0.0013528251019348093,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.0026476138480476066)}, 20: {'model_id': 20, 'rank': 2, 'cost': 0.44181954832983816, 'ensemble_weight': 0.02, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f141527b510>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1414f49f50>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1414fda910>, 'sklearn_regressor': LinearSVR(C=3987.8247542661084, dual=False, epsilon=0.0016864024742377878,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.003271167567550372)}, 27: {'model_id': 27, 'rank': 3, 'cost': 0.3965790528773321, 'ensemble_weight': 0.28, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1414e62f50>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f14147ce5d0>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1414f49650>, 'sklearn_regressor': LinearSVR(C=282.3574025132312, dual=False, epsilon=0.03693950282588793,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.010323108760937219)}, 29: {'model_id': 29, 'rank': 4, 'cost': 0.5123267487940523, 'ensemble_weight': 0.06, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f14146baad0>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1414ec4150>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f14147ce6d0>, 'sklearn_regressor': LinearSVR(C=5372.9271231497305, dual=False, epsilon=0.27641933499410104,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=0.0001200407109776716)}, 36: {'model_id': 36, 'rank': 5, 'cost': 0.39614153726559076, 'ensemble_weight': 0.22, 'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing.DataPreprocessorChoice object at 0x7f1414bd2f50>, 'feature_preprocessor': <autosklearn.pipeline.components.feature_preprocessing.FeaturePreprocessorChoice object at 0x7f1414672610>, 'regressor': <autosklearn.pipeline.components.regression.RegressorChoice object at 0x7f1414b0e310>, 'sklearn_regressor': LinearSVR(C=11.074793947670992, dual=False, epsilon=0.0020430206341446903,
#           loss='squared_epsilon_insensitive', random_state=42,
#           tol=1.8383594852137623e-05)}}