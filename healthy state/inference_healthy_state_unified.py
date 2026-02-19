import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# --- Configuration ---
BASE_DIR = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\healthy state'
DATA_PATH_HS = os.path.join(BASE_DIR, 'DS02_healthyStateInference.parquet')
OUTPUT_PATH_HS = os.path.join(BASE_DIR, 'inference_results.parquet')

DATA_PATH_TEST = os.path.join(BASE_DIR, 'DS02_healthyStateInference_TEST.parquet')
OUTPUT_PATH_TEST = os.path.join(BASE_DIR, 'inference_results_TEST.parquet')

SCALER_PATH = os.path.join(BASE_DIR, 'normalization_params.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Model Definition (Must match the notebook) ---
class GRUReconstructor(nn.Module):
    def __init__(self, input_dim=18, output_dim=14, dropout=0.2):
        super(GRUReconstructor, self).__init__()
        
        self.gru1 = nn.GRU(input_dim, 48, batch_first=True)
        self.gru2 = nn.GRU(48, 16, batch_first=True)
        self.gru3 = nn.GRU(16, 6, batch_first=True) # Reduced latent space to 6
        
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 24)
        self.fc3 = nn.Linear(24, output_dim) # Output dim is 14 (X only)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, 64, 18)
        
        # GRU Layers
        out, _ = self.gru1(x) # (batch, 64, 48)
        out, _ = self.gru2(out) # (batch, 64, 16)
        out, _ = self.gru3(out) # (batch, 64, 6)
        
        # Take only the last time step
        last_step = out[:, -1, :] # (batch, 6)
        
        # Dense Layers
        out = self.relu(self.fc1(last_step))
        out = self.relu(self.fc2(out))
        output = self.fc3(out) # Linear activation for output
        
        return output

# --- Dataset Class ---
class InferenceDataset(Dataset):
    def __init__(self, data_df, feature_columns, window_length=64):
        self.window_length = window_length
        self.feature_columns = feature_columns
        
        # Store data as a single numpy array
        self.data = data_df[feature_columns].values.astype(np.float32)
        
        # Calculate valid start indices
        self.valid_indices = []
        
        # Group by unit AND cycle to ensure windows don't cross cycle boundaries
        # Note: data_df must be sorted by unit and cycle and have a RangeIndex (0..N)
        grouped = data_df.groupby(['unit', 'cycle'])
        
        for _, group in grouped:
            group_len = len(group)
            if group_len >= window_length:
                # The indices of the group correspond to the indices in self.data
                # Valid start indices are from the first index of the group
                # up to (length - window_length)
                
                # group.index.values gives the global indices
                valid_starts = group.index.values[:group_len - window_length + 1]
                self.valid_indices.extend(valid_starts)
            
        self.valid_indices = np.array(self.valid_indices)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_length
        
        # Input: (64, 18)
        window = self.data[start_idx:end_idx].copy()
        
        # Target: (14,) - the last time step of the window, excluding W (first 4 columns)
        # The model is trained to reconstruct the last step of X given the sequence of W+X
        target = window[-1, 4:].copy() # Exclude W (alt, Mach, TRA, T2)
        
        return torch.from_numpy(window), torch.from_numpy(target), start_idx + self.window_length - 1

def run_inference(data_path, output_path):
    print(f"--- Processing {data_path} ---")
    
    # 1. Load Data
    print("Loading inference data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    df = df.sort_values(by=['unit', 'cycle']).reset_index(drop=True)
    
    # Define columns
    w_cols = ['alt', 'Mach', 'TRA', 'T2']
    # Filter out non-feature columns
    x_cols = [c for c in df.columns if c not in ['unit', 'cycle', 'hs'] + w_cols]
    
    # Feature columns for INPUT (W + X)
    feature_cols = w_cols + x_cols
    print(f"Feature columns: {feature_cols}")
    
    # 2. Normalize
    print("Normalizing data...")
    scaler = joblib.load(SCALER_PATH)
    
    # Create a copy for specific normalization
    df_normalized = df.copy()
    
    # Ensure all feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns in input data: {missing_cols}")
        return

    df_normalized[feature_cols] = scaler.transform(df[feature_cols])
    
    # 3. Load Model
    print("Loading model...")
    # Input dim: 18 (W+X), Output dim: 14 (X only)
    model = GRUReconstructor(input_dim=len(feature_cols), output_dim=len(x_cols)).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state dict: {e}")
        print("Please ensure the model architecture matches the saved state dict.")
        return

    model.eval()
    
    # 4. Prepare Dataset and DataLoader
    print("Preparing dataset...")
    window_length = 64
    dataset = InferenceDataset(df_normalized, feature_cols, window_length=window_length)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # 5. Run Inference
    print("Running inference...")
    all_predictions = []
    all_targets = []
    all_indices = []
    
    # Only run inference if dataset is not empty
    if len(dataset) > 0:
        with torch.no_grad():
            for inputs, targets, indices in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy()) # Targets are from the normalized df (X only)
                all_indices.append(indices.numpy())
                
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        indices = np.concatenate(all_indices, axis=0)
        
        # 6. Calculate Residuals
        # Residuals = |Target - Prediction|
        residuals = np.abs(targets - predictions)
    else:
        print("Warning: No valid windows found in the dataset!")
        predictions = np.array([])
        indices = np.array([])
        residuals = np.array([])
    
    # 7. Create Output DataFrame
    print("Creating output DataFrame...")
    
    # Initialize dictionary for new columns
    residual_dict = {}
    
    # Residuals are only for X columns
    for i, col in enumerate(x_cols):
        residual_name = f'R_{col}'
        # Create full-length arrays filled with NaN initially
        res_array = np.full(len(df), np.nan)
        
        if len(indices) > 0:
            # Fill in the calculated residuals at the correct indices
            res_array[indices] = residuals[:, i]
            
        residual_dict[residual_name] = res_array
        
    # Create result dataframe
    result_df = df[['unit', 'cycle', 'hs']].copy()
    
    # Add residual columns
    for name, data in residual_dict.items():
        result_df[name] = data

    # --- Fill Start of Cycles (First 63 steps of EACH CYCLE) ---
    print("Filling initial steps per cycle...")
    
    # Calculate indices dynamically based on window_length
    idx_first = window_length - 1
    idx_second = window_length
    
    # Group by unit AND cycle
    grouped_cycles = result_df.groupby(['unit', 'cycle'])
    
    for (unit_id, cycle_id), group in grouped_cycles:
        # Get indices for this group
        group_indices = group.index
        
        # If cycle has enough data to have valid predictions
        if len(group_indices) > idx_second: 
            
            # Global indices of the first two valid predictions in this cycle
            first_valid_idx_global = group_indices[idx_first]
            second_valid_idx_global = group_indices[idx_second]
            
            # Global indices to fill
            fill_indices_global = group_indices[:idx_first]
            
            for col_name in residual_dict.keys():
                val1 = result_df.at[first_valid_idx_global, col_name]
                val2 = result_df.at[second_valid_idx_global, col_name]
                avg_val = (val1 + val2) / 2.0
                
                result_df.loc[fill_indices_global, col_name] = avg_val
        else:
             # If cycle is too short (<= window_length), we have NO predictions.
             pass
    
    # 8. Save Results
    print(f"Saving results to {output_path}...")
    result_df.to_parquet(output_path)
    print("Done!")

if __name__ == "__main__":
    # Run for Healthy State Data
    run_inference(DATA_PATH_HS, OUTPUT_PATH_HS)
    
    print("-" * 50)
    
    # Run for Test Data
    run_inference(DATA_PATH_TEST, OUTPUT_PATH_TEST)
