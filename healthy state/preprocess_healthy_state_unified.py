import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import os

# Define paths
input_file = r"C:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\raw data\N-CMAPSS_DS02-006.h5"
output_dir = r"C:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\healthy state"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_dev_data():
    """
    Process Development set data (A_dev, W_dev, X_s_dev).
    Original logic from preprocess_healthy_state.py
    """
    print(f"\n[DEV] Loading data from {input_file}...")
    
    with h5py.File(input_file, 'r') as hdf:
        # Load Development set variables
        # Auxiliary info (unit, cycle, Fc, hs)
        A_dev = np.array(hdf.get('A_dev'))
        
        # A_var
        A_var = np.array(hdf.get('A_var'))
        A_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in A_var]
        
        # Operative conditions (W)
        W_dev = np.array(hdf.get('W_dev'))
        W_var = np.array(hdf.get('W_var'))
        W_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in W_var]
        
        # Sensor readings (Xs)
        X_s_dev = np.array(hdf.get('X_s_dev'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_s_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in X_s_var]
        
    print("[DEV] Data loaded. Processing full dataset...")
    
    # Define indices
    hs_index = 3
    unit_index = 0
    cycle_index = 1
    
    # Use full arrays
    A_dev_all = A_dev
    W_dev_all = W_dev
    X_s_dev_all = X_s_dev
    
    print(f"[DEV] Total samples: {len(A_dev_all)}")
    
    # Construct DataFrame with unit, cycle, hs, W, and Xs
    
    # 1. Unit, Cycle, and HS columns
    unit_col = A_dev_all[:, unit_index]
    cycle_col = A_dev_all[:, cycle_index]
    hs_col = A_dev_all[:, hs_index]
    
    # Create dictionaries for dataframe construction
    data_dict = {'unit': unit_col, 'cycle': cycle_col, 'hs': hs_col}
    
    # Add W columns
    for i, col_name in enumerate(W_var):
        data_dict[col_name] = W_dev_all[:, i]
        
    # Add Xs columns (before filtering)
    for i, col_name in enumerate(X_s_var):
        data_dict[col_name] = X_s_dev_all[:, i]
        
    df = pd.DataFrame(data_dict)
    
    print("[DEV] Applying Savitzky-Golay filter to Xs columns per unit and cycle...")
    
    # Define Savitzky-Golay parameters
    window_length = 9
    polyorder = 3
    
    # Columns to apply filter to (Xs only)
    xs_columns = X_s_var
    
    processed_dfs = []

    for (unit, cycle), group in df.groupby(['unit', 'cycle']):
        group = group.copy()
        
        # Apply filter to Xs columns
        for col in xs_columns:
             if len(group) >= window_length:
                group[col] = savgol_filter(group[col].values, window_length, polyorder)
        
        processed_dfs.append(group)
        
    df_processed = pd.concat(processed_dfs)
    
    # Define output files
    train_output_file = os.path.join(output_dir, "DS02_healthyStateTrain.parquet")
    test_output_file = os.path.join(output_dir, "DS02_healthyStateInference.parquet")
    
    # Save Train data (hs = 1)
    df_train = df_processed[df_processed['hs'] == 1].copy()
    print(f"[DEV] Saving healthy state training data (hs=1) to {train_output_file}...")
    print(f"[DEV] Train samples: {len(df_train)}")
    df_train.to_parquet(train_output_file, index=False)
    
    # Save Test data (all data)
    print(f"[DEV] Saving full test data (all hs) to {test_output_file}...")
    print(f"[DEV] Test samples: {len(df_processed)}")
    df_processed.to_parquet(test_output_file, index=False)
    
    print("[DEV] Done.")

def process_test_data():
    """
    Process Test set data (A_test, W_test, X_s_test).
    Original logic from preprocess_healthy_state_test.py
    """
    output_file = os.path.join(output_dir, "DS02_healthyStateInference_TEST.parquet")
    
    print(f"\n[TEST] Loading data from {input_file}...")
    
    with h5py.File(input_file, 'r') as hdf:
        # Load Test set variables
        # Auxiliary info (unit, cycle, Fc, hs)
        if 'A_test' in hdf:
            A_test = np.array(hdf.get('A_test'))
        elif 'Test' in hdf:
            A_test = np.array(hdf.get('Test').get('A'))
        else:
            raise KeyError("A_test or Test/A not found")

        # Variables names (var) are usually same for dev and test, or stored in root
        if 'A_var' in hdf:
             A_var = np.array(hdf.get('A_var'))
        else:
             # Fallback
             A_var = np.array(['unit', 'cycle', 'Fc', 'hs'])

        # Decode
        A_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in A_var]
        
        # Operative conditions (W)
        if 'W_test' in hdf:
            W_test = np.array(hdf.get('W_test'))
        elif 'Test' in hdf:
            W_test = np.array(hdf.get('Test').get('W'))
        
        if 'W_var' in hdf:
            W_var = np.array(hdf.get('W_var'))
        else:
            W_var = np.array(['alt', 'Mach', 'TRA', 'T2'])
            
        W_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in W_var]
        
        # Sensor readings (Xs)
        if 'X_s_test' in hdf:
            X_s_test = np.array(hdf.get('X_s_test'))
        elif 'Test' in hdf:
            X_s_test = np.array(hdf.get('Test').get('X_s'))

        if 'X_s_var' in hdf:
            X_s_var = np.array(hdf.get('X_s_var'))
        else:
            # Fallback list if needed
            X_s_var = np.array(['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf'])

        X_s_var = [x.decode('utf-8') if isinstance(x, bytes) else x for x in X_s_var]
        
    print("[TEST] Data loaded. Processing full TEST dataset...")
    
    # Define indices
    try:
        unit_index = 0
        cycle_index = 1
        # Check if 'hs' is in A_var to find index, else assume 3
        if 'hs' in A_var:
             hs_index = list(A_var).index('hs')
        else:
             hs_index = 3
    except:
        hs_index = 3

    print(f"[TEST] Total TEST samples: {len(A_test)}")
    print(f"[TEST] Features: A={A_var}, W={W_var}, Xs={X_s_var}")

    # Construct DataFrame
    
    # 1. Unit, Cycle, and HS columns
    unit_col = A_test[:, unit_index]
    cycle_col = A_test[:, cycle_index]
    hs_col = A_test[:, hs_index]
    
    data_dict = {'unit': unit_col, 'cycle': cycle_col, 'hs': hs_col}
    
    # Add W columns
    for i, col_name in enumerate(W_var):
        data_dict[col_name] = W_test[:, i]
        
    # Add Xs columns (before filtering)
    for i, col_name in enumerate(X_s_var):
        data_dict[col_name] = X_s_test[:, i]
        
    df = pd.DataFrame(data_dict)
    
    print("[TEST] Applying Savitzky-Golay filter to Xs columns per unit and cycle...")
    
    # Define Savitzky-Golay parameters
    window_length = 9
    polyorder = 3
    
    # Columns to apply filter to (Xs only)
    xs_columns = X_s_var
    
    processed_dfs = []

    # Group by unit and cycle
    for (unit, cycle), group in df.groupby(['unit', 'cycle']):
        group = group.copy()
        
        # Apply filter to Xs columns
        for col in xs_columns:
             if len(group) >= window_length:
                group[col] = savgol_filter(group[col].values, window_length, polyorder)
        
        processed_dfs.append(group)
        
    df_processed = pd.concat(processed_dfs)
    
    # Save Test data (all data)
    print(f"[TEST] Saving full TEST data for inference to {output_file}...")
    print(f"[TEST] Test samples: {len(df_processed)}")
    df_processed.to_parquet(output_file, index=False)
    
    print("[TEST] Done.")

def main():
    print("Starting combined preprocessing pipeline execution...")
    
    # 1. Process Development data
    process_dev_data()
    
    # 2. Process Test data
    process_test_data()
    
    print("\nAll preprocessing tasks completed successfully.")

if __name__ == "__main__":
    main()
