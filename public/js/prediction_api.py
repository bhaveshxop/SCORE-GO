from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS


# Initialize the Flask application

# Enable CORS for all routes

# Your existing routes and logic here


# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load the trained model
with open("cricket_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    try:
        data = request.get_json()

        # Create a DataFrame with feature names to match training data
        features = pd.DataFrame([{
            'runs': data['runs'],
            'wickets': data['wickets'],
            'overs': data['overs'],
            'target': data['target'],
            'run_rate': data['run_rate'],
            'remaining_runs': data['remaining_runs'],
            'remaining_overs': data['remaining_overs'],
            'required_run_rate': data['required_run_rate']
        }])

        # Get prediction
        prediction = model.predict(features)[0]

        # Return the prediction as JSON response
        return jsonify({'win_probability': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
