# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD THE TRAINED ARTIFACTS ---
# These are loaded only once when the server starts
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Check if the 'message' key exists
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
        
    message = data['message']
    
    # Transform the message using the loaded vectorizer
    message_vector = vectorizer.transform([message])
    
    # Make a prediction
    prediction_encoded = model.predict(message_vector)[0]
    
    # Decode the prediction back to the original label (e.g., 'financial')
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Return the result as JSON
    return jsonify({'prediction': prediction_label})

# A simple health check endpoint
@app.route('/')
def home():
    return "ML Model API is running!"

# This is needed for local testing


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)