from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os

app = Flask(__name__)

# Define the model architecture
class MultiStepZoneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_zones, output_steps=5, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_steps = output_steps
        self.zone_embedding = nn.Embedding(num_zones, 8)
        
        self.lstm = nn.LSTM(
            input_dim + 8,
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_steps)
        )
        
    def forward(self, x, zone_ids):
        embedded = self.zone_embedding(zone_ids).unsqueeze(1)
        embedded = embedded.expand(-1, x.size(1), -1)
        combined = torch.cat([x, embedded], dim=2)
        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Global variables to store loaded resources
global_model = None
global_scaler_X = None
global_scaler_y = None
global_zone_to_idx = None
global_device = None

# Load model and resources
def load_resources():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model settings
    model_path = "api/multi_step_zone_lstm_best.pth"
    scaler_X_path = "api/scaler_X.pkl"
    scaler_y_path = "api/scaler_y.pkl"
    zone_to_idx_path = "api/zone_to_idx.pkl"
    
    # Load artifacts
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    zone_to_idx = joblib.load(zone_to_idx_path)
    
    # Create model
    num_zones = len(zone_to_idx)
    input_dim = 7  # Number of numerical features
    hidden_dim = 128
    model = MultiStepZoneLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_zones=num_zones,
        output_steps=5,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, scaler_X, scaler_y, zone_to_idx, device

# Prepare input data for prediction
def prepare_data(data, seq_length=10):
    # Handle cyclical month feature
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
    # Get numerical features
    numerical_cols = [
        'Year', 'Temperature (°C)', 'Humidity (%)', 
        'Rainfall (mm)', 'Sunlight (hrs/day)', 'Month_sin', 'Month_cos'
    ]
    
    # Scale numerical features
    numerical_data = global_scaler_X.transform(data[numerical_cols].tail(seq_length))
    
    # Get zone ID
    zone_name = data['Zone'].iloc[-1]
    zone_id = global_zone_to_idx.get(zone_name)
    
    if zone_id is None:
        raise ValueError(f"Zone '{zone_name}' not found in the training data")
    
    return numerical_data, zone_id

# Make predictions
def get_predictions(input_data, zone_id):
    # Prepare input tensor
    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(global_device)
    zone_id = torch.tensor([zone_id], dtype=torch.long).to(global_device)
    
    # Get predictions
    with torch.no_grad():
        output = global_model(x, zone_id)
    
    # Convert to numpy and inverse transform
    predictions = output.cpu().numpy()
    predictions_unscaled = global_scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions_unscaled

# Generate future dates
def generate_future_dates(last_date, steps=5):
    future_dates = []
    current_year = last_date['Year']
    current_month = last_date['Month']
    
    for i in range(steps):
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        future_dates.append({'Year': int(current_year), 'Month': int(current_month)})
    
    return future_dates

# Initialize the model and resources when the app starts
# This replaces the deprecated before_first_request
with app.app_context():
    global_model, global_scaler_X, global_scaler_y, global_zone_to_idx, global_device = load_resources()
    print("Model and resources loaded successfully in app context")

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        json_data = request.get_json()
        
        # Convert to DataFrame
        data = pd.DataFrame(json_data['data'])
        
        # Check if we have enough data
        seq_length = 10
        if len(data) < seq_length:
            return jsonify({'error': f'Input data must contain at least {seq_length} records'}), 400
        
        # Prepare data and get predictions
        input_data, zone_id = prepare_data(data, seq_length)
        predictions = get_predictions(input_data, zone_id)
        
        # Generate future dates
        last_date = data.iloc[-1][['Year', 'Month']]
        future_dates = generate_future_dates(last_date)
        
        # Create response
        results = []
        for i, date in enumerate(future_dates):
            results.append({
                'Year': date['Year'],
                'Month': date['Month'],
                'Zone': data['Zone'].iloc[-1],
                'Predicted_Mycotoxin_ppb': float(predictions[i])
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Simple route to check if API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Mycotoxin Prediction API is running. Send POST requests to /predict'
    })

if __name__ == '__main__':
    # Alternative way to load resources at startup if app_context doesn't work
    if global_model is None:
        global_model, global_scaler_X, global_scaler_y, global_zone_to_idx, global_device = load_resources()
        print("Model and resources loaded successfully at startup")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)