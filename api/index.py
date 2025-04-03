# from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import joblib
# import os

# app = Flask(__name__)

# # Define the model architecture
# class MultiStepZoneLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_zones, output_steps=5, num_layers=2, dropout=0.2):
#         super().__init__()
#         self.output_steps = output_steps
#         self.zone_embedding = nn.Embedding(num_zones, 8)
        
#         self.lstm = nn.LSTM(
#             input_dim + 8,
#             hidden_dim, 
#             num_layers, 
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, output_steps)
#         )
        
#     def forward(self, x, zone_ids):
#         embedded = self.zone_embedding(zone_ids).unsqueeze(1)
#         embedded = embedded.expand(-1, x.size(1), -1)
#         combined = torch.cat([x, embedded], dim=2)
#         lstm_out, _ = self.lstm(combined)
#         out = self.fc(lstm_out[:, -1, :])
#         return out

# # Global variables to store loaded resources
# global_model = None
# global_scaler_X = None
# global_scaler_y = None
# global_zone_to_idx = None
# global_device = None

# # Load model and resources
# def load_resources():
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Model settings
#     model_path = "api/multi_step_zone_lstm_best.pth"
#     scaler_X_path = "api/scaler_X.pkl"
#     scaler_y_path = "api/scaler_y.pkl"
#     zone_to_idx_path = "api/zone_to_idx.pkl"
    
#     # Load artifacts
#     scaler_X = joblib.load(scaler_X_path)
#     scaler_y = joblib.load(scaler_y_path)
#     zone_to_idx = joblib.load(zone_to_idx_path)
    
#     # Create model
#     num_zones = len(zone_to_idx)
#     input_dim = 7  # Number of numerical features
#     hidden_dim = 128
#     model = MultiStepZoneLSTM(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         num_zones=num_zones,
#         output_steps=5,
#         num_layers=2,
#         dropout=0.3
#     ).to(device)
    
#     # Load model weights
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     return model, scaler_X, scaler_y, zone_to_idx, device

# # Prepare input data for prediction
# def prepare_data(data, seq_length=10):
#     # Handle cyclical month feature
#     data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
#     data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
#     # Get numerical features
#     numerical_cols = [
#         'Year', 'Temperature (°C)', 'Humidity (%)', 
#         'Rainfall (mm)', 'Sunlight (hrs/day)', 'Month_sin', 'Month_cos'
#     ]
    
#     # Scale numerical features
#     numerical_data = global_scaler_X.transform(data[numerical_cols].tail(seq_length))
    
#     # Get zone ID
#     zone_name = data['Zone'].iloc[-1]
#     zone_id = global_zone_to_idx.get(zone_name)
    
#     if zone_id is None:
#         raise ValueError(f"Zone '{zone_name}' not found in the training data")
    
#     return numerical_data, zone_id

# # Make predictions
# def get_predictions(input_data, zone_id):
#     # Prepare input tensor
#     x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(global_device)
#     zone_id = torch.tensor([zone_id], dtype=torch.long).to(global_device)
    
#     # Get predictions
#     with torch.no_grad():
#         output = global_model(x, zone_id)
    
#     # Convert to numpy and inverse transform
#     predictions = output.cpu().numpy()
#     predictions_unscaled = global_scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
#     return predictions_unscaled

# # Generate future dates
# def generate_future_dates(last_date, steps=5):
#     future_dates = []
#     current_year = last_date['Year']
#     current_month = last_date['Month']
    
#     for i in range(steps):
#         current_month += 1
#         if current_month > 12:
#             current_month = 1
#             current_year += 1
#         future_dates.append({'Year': int(current_year), 'Month': int(current_month)})
    
#     return future_dates

# # Initialize the model and resources when the app starts
# # This replaces the deprecated before_first_request
# with app.app_context():
#     global_model, global_scaler_X, global_scaler_y, global_zone_to_idx, global_device = load_resources()
#     print("Model and resources loaded successfully in app context")

# # API endpoint for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data
#         json_data = request.get_json()
        
#         # Convert to DataFrame
#         data = pd.DataFrame(json_data['data'])
        
#         # Check if we have enough data
#         seq_length = 10
#         if len(data) < seq_length:
#             return jsonify({'error': f'Input data must contain at least {seq_length} records'}), 400
        
#         # Prepare data and get predictions
#         input_data, zone_id = prepare_data(data, seq_length)
#         predictions = get_predictions(input_data, zone_id)
        
#         # Generate future dates
#         last_date = data.iloc[-1][['Year', 'Month']]
#         future_dates = generate_future_dates(last_date)
        
#         # Create response
#         results = []
#         for i, date in enumerate(future_dates):
#             results.append({
#                 'Year': date['Year'],
#                 'Month': date['Month'],
#                 'Zone': data['Zone'].iloc[-1],
#                 'Predicted_Mycotoxin_ppb': float(predictions[i])
#             })
        
#         return jsonify({'predictions': results})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Simple route to check if API is running
# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         'status': 'online',
#         'message': 'Mycotoxin Prediction API is running. Send POST requests to /predict'
#     })

# if __name__ == '__main__':
#     # Alternative way to load resources at startup if app_context doesn't work
#     if global_model is None:
#         global_model, global_scaler_X, global_scaler_y, global_zone_to_idx, global_device = load_resources()
#         print("Model and resources loaded successfully at startup")
    
#     # Run Flask app
#     app.run(host='0.0.0.0', port=5000, debug=True)

# app_onnx.py (or api/index.py for Vercel)
import flask
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import onnxruntime as ort # Import ONNX runtime

app = Flask(__name__)

# --- Global variables to store loaded resources ---
global_onnx_session = None # Stores the ONNX inference session
global_scaler_X = None     # Stores the feature scaler
global_scaler_y = None     # Stores the target scaler
global_zone_to_idx = None  # Stores the zone mapping

# --- Load ONNX model and other resources ---
def load_resources():
    """Loads scalers, zone mapping, and the ONNX inference session."""
    print("Loading resources for ONNX inference...")

    # Define paths to artifacts within the 'api' directory
    onnx_model_path = "api/mycotoxin_model.onnx"
    scaler_X_path = "api/scaler_X.pkl"
    scaler_y_path = "api/scaler_y.pkl"
    zone_to_idx_path = "api/zone_to_idx.pkl"

    # Check if files exist
    for path in [onnx_model_path, scaler_X_path, scaler_y_path, zone_to_idx_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Required file not found at {path}")

    # Load scalers and zone mapping using joblib
    print("Loading scalers and zone mapping...")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    zone_to_idx = joblib.load(zone_to_idx_path)
    print("Scalers and zone mapping loaded.")

    # Load the ONNX model and create an inference session
    print(f"Loading ONNX model from {onnx_model_path}...")
    try:
        # Use CPUExecutionProvider as Vercel runs on CPU
        onnx_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        print(f"ONNX model loaded successfully. Input names: {onnx_session.get_inputs()[0].name}, {onnx_session.get_inputs()[1].name}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        raise RuntimeError(f"Failed to load ONNX model from {onnx_model_path}") from e

    # Return loaded resources
    return onnx_session, scaler_X, scaler_y, zone_to_idx

# --- Prepare input data for prediction ---
def prepare_data(data_df, seq_length=10):
    """Prepares the input DataFrame for prediction."""
    print(f"Preparing data (last {seq_length} rows)...")
    # Ensure data has expected columns
    required_cols = {'Year', 'Month', 'Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Sunlight (hrs/day)', 'Zone'}
    if not required_cols.issubset(data_df.columns):
        raise ValueError(f"Input data missing required columns. Found: {list(data_df.columns)}, Required: {list(required_cols)}")

    # Handle cyclical month feature
    data_df['Month_sin'] = np.sin(2 * np.pi * data_df['Month'] / 12)
    data_df['Month_cos'] = np.cos(2 * np.pi * data_df['Month'] / 12)

    # Define numerical feature columns in the correct order for scaling
    numerical_cols = [
        'Year', 'Temperature (°C)', 'Humidity (%)',
        'Rainfall (mm)', 'Sunlight (hrs/day)', 'Month_sin', 'Month_cos'
    ]

    # Select the last 'seq_length' rows and scale numerical features
    numerical_data = global_scaler_X.transform(data_df[numerical_cols].tail(seq_length))

    # Get zone name and map it to an index
    zone_name = data_df['Zone'].iloc[-1]
    zone_id = global_zone_to_idx.get(zone_name)

    if zone_id is None:
        # Log available zones for easier debugging
        # print(f"Available zones: {list(global_zone_to_idx.keys())}")
        raise ValueError(f"Zone '{zone_name}' not found in the training data mapping.")

    print(f"Data prepared. Shape: {numerical_data.shape}, Zone: {zone_name} (ID: {zone_id})")
    # Return NumPy array and integer zone ID
    return numerical_data, zone_id

# --- Make predictions using ONNX session ---
def get_predictions_onnx(input_data_np, zone_id_int):
    """Runs inference using the loaded ONNX session."""
    print("Running ONNX inference...")
    # ONNX Runtime expects inputs as a dictionary:
    # Keys must match the 'input_names' defined during ONNX export
    # Values must be NumPy arrays with the correct data types

    # Ensure input_data_np has batch dimension [1, seq_length, features]
    if input_data_np.ndim == 2:
         input_data_np = np.expand_dims(input_data_np, axis=0)

    # Ensure correct data types (float32 for sequence, int64 for zone ID)
    input_sequence_np = input_data_np.astype(np.float32)
    # Zone ID needs to be a NumPy array as well, shape (1,) for batch size 1
    input_zone_id_np = np.array([zone_id_int], dtype=np.int64)

    ort_inputs = {
        global_onnx_session.get_inputs()[0].name: input_sequence_np, # e.g., "input_sequence"
        global_onnx_session.get_inputs()[1].name: input_zone_id_np   # e.g., "input_zone_id"
    }
    print(f"ONNX input shapes: sequence={input_sequence_np.shape}, zone_id={input_zone_id_np.shape}")

    # Run inference, requesting the output node name defined during export
    output_names = [global_onnx_session.get_outputs()[0].name] # e.g., ["predictions"]
    ort_outputs = global_onnx_session.run(output_names, ort_inputs)

    # The output is a list containing the predictions array
    predictions_np = ort_outputs[0]
    print(f"ONNX output shape: {predictions_np.shape}") # Should be [1, output_steps]

    # Inverse transform the scaled predictions
    # Scaler expects shape [n_samples, n_features], predictions_np is [1, output_steps]
    # Reshape if necessary, flatten after inverse transform
    predictions_unscaled = global_scaler_y.inverse_transform(predictions_np).flatten()
    print("Predictions unscaled.")

    return predictions_unscaled

# --- Generate future dates ---
def generate_future_dates(last_date_series, steps=5):
    """Generates dicts for future year/month based on the last date."""
    future_dates = []
    # Ensure types are standard Python int
    current_year = int(last_date_series['Year'])
    current_month = int(last_date_series['Month'])

    for _ in range(steps):
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        future_dates.append({'Year': current_year, 'Month': current_month})
    return future_dates

# --- Initialize resources at application startup ---
# Using Flask's app_context for robust initialization
print("Attempting to load resources within app context...")
with app.app_context():
    try:
        global_onnx_session, global_scaler_X, global_scaler_y, global_zone_to_idx = load_resources()
        print("ONNX session and resources loaded successfully in app context.")
    except Exception as e:
        # Log the error during startup
        app.logger.error(f"FATAL: Failed to load resources during startup: {e}", exc_info=True)
        # Depending on severity, you might want the app to fail fast
        # raise RuntimeError(f"Application failed to initialize: {e}") from e
        print(f"FATAL ERROR during resource loading: {e}") # Print to console as well

# --- API endpoint for prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST requests to make mycotoxin predictions."""
    # Ensure resources are loaded before handling request
    if global_onnx_session is None:
         app.logger.error("Prediction endpoint called but ONNX session is not loaded.")
         return jsonify({'error': 'Model not ready, initialization may have failed.'}), 503 # Service Unavailable

    print("\nReceived prediction request.")
    try:
        # Get JSON data from the request
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
             return jsonify({'error': 'Invalid request format. Missing "data" key.'}), 400

        # Convert input JSON data to DataFrame
        data = pd.DataFrame(json_data['data'])
        print(f"Input data contains {len(data)} records.")

        # Check if we have enough historical data
        seq_length = 10 # Input sequence length
        if len(data) < seq_length:
            return jsonify({'error': f'Input data must contain at least {seq_length} records for history.'}), 400

        # Prepare data: scale features, get zone ID
        # Returns NumPy array (features) and integer (zone_id)
        input_features_np, zone_id = prepare_data(data, seq_length)

        # Get predictions using the ONNX model
        predictions = get_predictions_onnx(input_features_np, zone_id)

        # Generate future dates based on the last record in the input
        last_date = data.iloc[-1][['Year', 'Month']]
        future_dates = generate_future_dates(last_date, steps=len(predictions))

        # Create the response structure
        results = []
        zone_name = data['Zone'].iloc[-1] # Get zone name for response
        for i, date_info in enumerate(future_dates):
            results.append({
                'Year': date_info['Year'],
                'Month': date_info['Month'],
                'Zone': zone_name,
                # Ensure prediction value is a standard float for JSON serialization
                'Predicted_Mycotoxin_ppb': float(predictions[i])
            })

        print("Prediction successful. Returning results.")
        return jsonify({'predictions': results})

    except ValueError as ve:
        # Handle specific data-related errors (e.g., missing columns, unknown zone)
        app.logger.warning(f"Data validation error: {ve}")
        return jsonify({'error': f'Data Error: {str(ve)}'}), 400
    except FileNotFoundError as fnf:
        # Handle case where model/scaler files missing after startup
        app.logger.error(f"Missing resource file during prediction: {fnf}")
        return jsonify({'error': 'Server configuration error: Missing model resources.'}), 500
    except Exception as e:
        # Log unexpected errors for debugging
        app.logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        # Return a generic server error to the client
        return jsonify({'error': 'An internal server error occurred.'}), 500

# --- Simple health check / home route ---
@app.route('/', methods=['GET'])
def home():
    """Provides a simple status check."""
    status = 'online' if global_onnx_session else 'offline (initializing or failed)'
    return jsonify({
        'status': status,
        'message': 'Mycotoxin Prediction API (ONNX Runtime). Send POST requests to /predict'
    })

# --- Main execution block (for local testing) ---
if __name__ == '__main__':
    # Ensure resources are loaded if running directly (fallback)
    if global_onnx_session is None:
        try:
            print("Loading resources directly (not in app context)...")
            global_onnx_session, global_scaler_X, global_scaler_y, global_zone_to_idx = load_resources()
            print("ONNX session and resources loaded successfully at startup.")
        except Exception as e:
            print(f"FATAL ERROR during direct resource loading: {e}")
            # Exit if resources can't be loaded when run directly
            exit(1)

    # Run Flask development server (set debug=False for production/Vercel)
    # Vercel uses its own server (e.g., waitress or gunicorn implicitly)
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Use debug=False for Vercel