import pandas as pd
import numpy as np

def generate_cricket_dataset(n_samples=10000):
    np.random.seed(42)

    data = {
        'runs': np.random.randint(0, 350, size=n_samples),
        'wickets': np.random.randint(0, 10, size=n_samples),
        'overs': np.random.uniform(0.1, 50, size=n_samples),
        'target': np.random.randint(50, 400, size=n_samples),
    }
    data['run_rate'] = np.round(data['runs'] / data['overs'], 2)  # Realistic run rate
    data['remaining_runs'] = data['target'] - data['runs']
    data['remaining_overs'] = np.round(50 - data['overs'], 2)

    # Add features for better realism
    data['required_run_rate'] = np.round(
        np.where(data['remaining_runs'] > 0, data['remaining_runs'] / (data['remaining_overs'] + 0.1), 0), 2
    )
    data['win_probability'] = np.where(
        (data['runs'] >= data['target']) | (data['remaining_overs'] <= 0), 1,
        np.where(data['remaining_runs'] > 0, 1 - (data['remaining_runs'] / data['target']), 0)
    )
    data['win_probability'] = np.clip(data['win_probability'], 0, 1)  # Ensure probabilities are between 0 and 1

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate the dataset
    data = generate_cricket_dataset()
    data.to_csv("cricket_dataset.csv", index=False)
    print("Dataset created and saved to cricket_dataset.csv")
