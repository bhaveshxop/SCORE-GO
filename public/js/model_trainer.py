import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv("realistic_cricket_dataset.csv")

X = data[['runs', 'wickets', 'overs', 'target', 'run_rate', 'remaining_runs', 'remaining_overs', 'required_run_rate']]
y = data['win_probability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5]
}

base_model = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

print("Training model with GridSearch...")
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_


train_predictions = best_model.predict(X_train_scaled)
test_predictions = best_model.predict(X_test_scaled)

train_predictions = np.clip(train_predictions, 0, 1)
test_predictions = np.clip(test_predictions, 0, 1)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("\nBest Parameters:", grid_search.best_params_)
print("\nTraining MSE:", train_mse)
print("Training R2:", train_r2)
print("Test MSE:", test_mse)
print("Test R2:", test_r2)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': list(X.columns),
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance.to_dict()
}

with open("cricket_model.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)
print("\nModel and artifacts saved to cricket_model.pkl")