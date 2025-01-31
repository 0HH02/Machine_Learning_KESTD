import pandas as pd
import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

file_path = "DB/output_for_automl.csv"  
data = pd.read_csv(file_path)

threshold = 0.8
cols_to_keep = data.columns[data.isnull().mean() < threshold]
data = data[cols_to_keep]

min_val = data.min().min() 
placeholder_value = min_val - 9964.995
data.fillna(placeholder_value, inplace=True) 

# print(min_val) -35.005
# print(max_val) 2854.0

X = data.drop(columns=['007_score_ligando'])
y = data['007_score_ligando']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,  
    per_run_time_limit=30, 
    ensemble_kwargs={'ensemble_size': 50}, 
    seed=42
)

print("Entrenando Auto-sklearn...")
automl.fit(X_train, y_train)

y_pred = automl.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

print("\nRanking de los modelos usados:")
print(automl.leaderboard())

print("\nMejor modelo:")
print(automl.show_models())

joblib.dump(automl, "x_automl_model.pkl")
