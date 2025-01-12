import pandas as pd
import numpy as np

def generate_cricket_dataset(n_samples=10000):
    np.random.seed(42)
    
    overs = np.random.uniform(5, 50, size=n_samples)
    

    base_target = np.where(
        overs < 20,
        np.random.normal(150, 20, n_samples),  # T20-style targets
        np.random.normal(280, 40, n_samples)   # ODI-style targets
    )
    target = np.clip(base_target, 120, 400).astype(int)
   
    innings_progression = overs / 50
    
    powerplay_factor = np.where(overs <= 10, 1.2, 1.0)
    middle_overs_factor = np.where((overs > 10) & (overs <= 40), 0.9, 1.0)
    death_overs_factor = np.where(overs > 40, 1.3, 1.0)
    phase_factor = powerplay_factor * middle_overs_factor * death_overs_factor
    

    runs = target * (overs / 50) * phase_factor
    runs = np.clip(runs, 0, target)
    
  
    wicket_probability = np.clip(
        0.5 * innings_progression + 
        0.3 * np.random.random(n_samples) +
        0.2 * (runs / target),
        0, 1  # Ensure probability is between 0 and 1
    )
    wickets = np.random.binomial(10, wicket_probability)
    
   
    wicket_impact = np.power((10 - wickets) / 10, 1.5)
    runs = runs * wicket_impact
    
    remaining_overs = np.round(50 - overs, 2)
    remaining_overs = np.clip(remaining_overs, 0, 50)  # Ensure non-negative
    remaining_runs = target - runs
    remaining_runs = np.clip(remaining_runs, 0, target)  # Ensure non-negative
    
    run_rate = np.round(np.where(overs > 0, runs / overs, 0), 2)
    required_run_rate = np.round(
        np.where(remaining_overs > 0, remaining_runs / remaining_overs, 999), 2
    )
    
    rate_difference = required_run_rate - run_rate
    acceleration_factor = np.clip(1 + (rate_difference / 10), 0.8, 1.5)
    runs = runs * acceleration_factor
    
    # Final runs cleanup
    runs = np.clip(runs, 0, target)
    runs = np.round(runs).astype(int)
    
    # Recalculate rates after adjustments
    remaining_runs = target - runs
    run_rate = np.round(np.where(overs > 0, runs / overs, 0), 2)
    required_run_rate = np.round(
        np.where(remaining_overs > 0, remaining_runs / remaining_overs, 999), 2
    )
    
    # Enhanced win probability calculation
    win_factors = (
        -0.01 * (runs - (target * innings_progression)) +  # Progress vs expected
        0.4 * np.clip(required_run_rate - run_rate, -5, 5) +  # Rate pressure
        0.3 * wickets +                                    # Wicket impact
        0.2 * (remaining_overs / 50) +                     # Game stage
        -0.3 * (run_rate > required_run_rate) +           # Bonus for being ahead of rate
        0.2 * (wickets >= 7) +                            # Extra penalty for 7+ wickets
        -0.2 * (overs <= 10) * (run_rate > 6)            # Bonus for good powerplay
    )
    
    win_probability = 1 / (1 + np.exp(win_factors))
    
    # Special case adjustments for win probability
    win_probability = np.where(runs >= target, 1, win_probability)  # Won
    win_probability = np.where(wickets == 10, 0, win_probability)   # All out
    win_probability = np.where(
        (remaining_overs <= 0) & (runs < target), 
        0, 
        win_probability
    )
    
    # Higher probability if ahead of required rate
    win_probability = np.where(
        (required_run_rate < run_rate) & (wickets <= 5) & (remaining_overs >= 10),
        np.clip(win_probability * 1.3, 0, 1),
        win_probability
    )
    
    # Lower probability if behind required rate
    win_probability = np.where(
        required_run_rate > (run_rate + 4),
        np.clip(win_probability * 0.7, 0, 1),
        win_probability
    )
    
    win_probability = np.round(np.clip(win_probability, 0, 1), 3)
    
    # Create final dataset
    data = {
        'runs': runs,
        'wickets': wickets,
        'overs': overs,
        'target': target,
        'run_rate': run_rate,
        'remaining_runs': remaining_runs,
        'remaining_overs': remaining_overs,
        'required_run_rate': required_run_rate,
        'win_probability': win_probability
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate the dataset
    data = generate_cricket_dataset()
    data.to_csv("realistic_cricket_dataset.csv", index=False)
    print("Dataset created and saved to realistic_cricket_dataset.csv")