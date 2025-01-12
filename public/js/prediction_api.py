from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

with open("cricket_model.pkl", "rb") as f:
    model_artifacts = pickle.load(f)

model = model_artifacts['model']
scaler = model_artifacts['scaler']

@app.route('/predict', methods=['POST'])
def predict():
    try:
     
        data = request.get_json()
        print("Received data:", data)

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
        
    
        features = features[model_artifacts['feature_names']]

        features_scaled = scaler.transform(features)

    
        prediction = model.predict(features_scaled)[0]
        
        prediction = np.clip(prediction, 0, 1)
        
        prediction = round(float(prediction), 3)

        response = {
            'win_probability': prediction,
            'status': 'success',
            'message': 'Prediction successful'
        }
        
        print("Response:", response)
        return jsonify(response)
    
    except KeyError as e:
        error_msg = f"Missing required feature: {str(e)}"
        print("Error:", error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': 'KeyError'
        }), 400
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print("Error:", error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'error_type': type(e).__name__
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running'
    })

if __name__ == '__main__':
    print("Model loaded successfully")
    print("Feature names:", model_artifacts['feature_names'])
    print("Server starting...")
    app.run(debug=True)