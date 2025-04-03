# convert_to_onnx.py
import torch
import torch.nn as nn
import joblib
import os
import numpy as np # Need numpy for prepare_data if used for num_zones source
import pandas as pd # Need pandas if reading data for num_zones source

print(f"PyTorch version: {torch.__version__}")
try:
    import onnx
    print(f"ONNX version: {onnx.__version__}")
except ImportError:
    print("ONNX library not found. Please install with: pip install onnx")
    exit()

# --- Define the PyTorch Model Architecture ---
# (Paste the exact class definition you used for training)
class MultiStepZoneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_zones, output_steps=5, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_steps = output_steps
        self.zone_embedding = nn.Embedding(num_zones, 8)

        self.lstm = nn.LSTM(
            input_dim + 8, # Embedding size added
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
        # x shape: [batch_size, seq_len, input_dim]
        # zone_ids shape: [batch_size]
        embedded = self.zone_embedding(zone_ids) # shape: [batch_size, embedding_dim]
        embedded = embedded.unsqueeze(1) # shape: [batch_size, 1, embedding_dim]
        embedded = embedded.expand(-1, x.size(1), -1) # shape: [batch_size, seq_len, embedding_dim]

        combined = torch.cat([x, embedded], dim=2) # shape: [batch_size, seq_len, input_dim + embedding_dim]
        lstm_out, _ = self.lstm(combined) # shape: [batch_size, seq_len, hidden_dim]

        # Use the output of the last time step for prediction
        last_time_step_out = lstm_out[:, -1, :] # shape: [batch_size, hidden_dim]
        out = self.fc(last_time_step_out) # shape: [batch_size, output_steps]
        return out

# --- Configuration ---
# !! ADJUST THESE PATHS !!
pytorch_model_path = "multi_step_zone_lstm_best.pth"
zone_to_idx_path = "zone_to_idx.pkl"
# !! SET OUTPUT PATH !!
onnx_model_path = "mycotoxin_model.onnx"

# Model parameters (must match the saved model structure exactly)
input_dim = 7       # Number of numerical input features (Year, Temp, Hum, Rain, Sun, Month_sin, Month_cos)
hidden_dim = 128    # LSTM hidden dimension
output_steps = 5    # Number of future steps to predict
num_layers = 2      # Number of LSTM layers
dropout = 0.3       # Dropout rate used in training
seq_length = 10     # Input sequence length expected by the model

# --- Load necessary info ---
print("Loading zone mapping...")
if not os.path.exists(zone_to_idx_path):
    print(f"Error: Zone mapping file not found at {zone_to_idx_path}")
    exit()
zone_to_idx = joblib.load(zone_to_idx_path)
num_zones = len(zone_to_idx)
print(f"Number of zones found: {num_zones}")

# Use CPU for conversion process
device = torch.device("cpu")

# --- Instantiate PyTorch Model ---
print("Instantiating PyTorch model...")
model = MultiStepZoneLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_zones=num_zones,
    output_steps=output_steps,
    num_layers=num_layers,
    dropout=dropout
).to(device)

# --- Load Trained Weights ---
print(f"Loading trained weights from {pytorch_model_path}...")
if not os.path.exists(pytorch_model_path):
    print(f"Error: PyTorch model file not found at {pytorch_model_path}")
    exit()
model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
model.eval() # Set to evaluation mode! Crucial for dropout/batchnorm layers
print("Weights loaded successfully.")

# --- Create Dummy Input Data ---
# Create example tensors matching the expected input shapes and types
# We use batch size 1, but make it dynamic during export
print("Creating dummy input data...")
dummy_x = torch.randn(1, seq_length, input_dim, dtype=torch.float32).to(device)
# Ensure dummy zone ID is within the valid range [0, num_zones-1]
dummy_zone_id = torch.randint(0, num_zones, (1,), dtype=torch.long).to(device)

# Store inputs in the order the model's forward() method expects them
dummy_input_tuple = (dummy_x, dummy_zone_id)
print(f"Dummy input shapes: x={dummy_x.shape}, zone_id={dummy_zone_id.shape}")

# --- Define Input/Output Names and Dynamic Axes ---
# These names will be used to feed data into the ONNX model later
input_names = ["input_sequence", "input_zone_id"]
output_names = ["predictions"] # Name of the output node

# Define which axes can have variable sizes.
# We make the batch size (axis 0) dynamic for all inputs and outputs.
dynamic_axes = {
    input_names[0]: {0: "batch_size"},  # batch_size is dynamic for input_sequence
    input_names[1]: {0: "batch_size"},  # batch_size is dynamic for input_zone_id
    output_names[0]: {0: "batch_size"}  # batch_size is dynamic for predictions
}
print(f"Input names: {input_names}")
print(f"Output names: {output_names}")
print(f"Dynamic axes: {dynamic_axes}")

# --- Export to ONNX ---
print(f"Attempting to export model to ONNX format at {onnx_model_path}...")
try:
    torch.onnx.export(
        model,                      # The PyTorch model instance
        dummy_input_tuple,          # Tuple of dummy inputs
        onnx_model_path,            # Output path for the ONNX file
        export_params=True,         # Store trained weights within the ONNX file
        opset_version=11,           # ONNX version to use (11+ recommended)
        do_constant_folding=True,   # Apply basic optimizations
        input_names=input_names,    # Assign names to input nodes
        output_names=output_names,  # Assign names to output nodes
        dynamic_axes=dynamic_axes   # Specify dynamic dimensions
    )
    print("-" * 30)
    print("ONNX model exported successfully!")
    print(f"Model saved to: {os.path.abspath(onnx_model_path)}")
    print("-" * 30)
    # Optional: Verify the ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful.")

except Exception as e:
    print("\n" + "="*30)
    print("Error during ONNX export:")
    print(e)
    print("="*30 + "\n")
    print("Troubleshooting Tips:")
    print("- Ensure PyTorch, ONNX libraries are up-to-date.")
    print("- Check if all model parameters match the loaded state_dict.")
    print("- Verify dummy input shapes and types exactly match model's forward method.")
    print("- Try a different opset_version (e.g., 12, 13).")
    print("- Simplify the model temporarily to isolate problematic operations.")