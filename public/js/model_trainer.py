import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("cricket_dataset.csv")
    
    # Define features (X) and target (y)
    X = data[['runs', 'wickets', 'overs', 'target', 'run_rate', 'remaining_runs', 'remaining_overs', 'required_run_rate']]
    y = data['win_probability']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use GradientBoostingRegressor for continuous target variable
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Save the trained model to a file
    with open("cricket_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to cricket_model.pkl")
