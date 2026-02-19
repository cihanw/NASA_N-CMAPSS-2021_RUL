import h5py
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import gc

def process_dev_data(chunk_size=1000000):
    """
    Process Development set data (A_dev, W_dev, X_s_dev) for RUL prediction.
    Logic from RUL/Mamba2_preprocess.py
    """
    # File paths
    h5_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\raw data\N-CMAPSS_DS02-006.h5'
    parquet_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\healthy state\inference_results.parquet'
    output_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\RUL\mamba2_processed.parquet'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[DEV] Processing RUL data in chunks of {chunk_size}...")

    # --- 0. PRE-VALIDATION ---
    print("[DEV] Checking file lengths...")
    with h5py.File(h5_path, 'r') as hf:
        if 'A_dev' in hf:
            h5_rows = hf['A_dev'].shape[0]
        elif 'Dev' in hf:
            h5_rows = hf['Dev']['A'].shape[0]
        else:
            raise KeyError("A_dev or Dev/A not found in H5")
            
    try:
        # Read only unit column to get length quickly
        pq_meta = pd.read_parquet(parquet_path, columns=['unit']) 
        pq_rows = len(pq_meta)
        del pq_meta
    except Exception as e:
        raise RuntimeError(f"Could not read Parquet file: {e}")
        
    print(f"[DEV] H5 Rows: {h5_rows}, Parquet Rows: {pq_rows}")
    
    if h5_rows != pq_rows:
        diff = abs(h5_rows - pq_rows)
        if diff > 1000:
             raise ValueError(f"CRITICAL: Row count mismatch is too large ({diff}). H5={h5_rows}, Parquet={pq_rows}. Alignment is likely impossible.")
        else:
             print(f"WARNING: Row count mismatch ({diff}). Will attempt to pad/truncate, but alignment is not guaranteed.")
    else:
        print("[DEV] Row counts match perfectly.")

    # Open H5 for reading
    with h5py.File(h5_path, 'r') as hf:
        # Check keys and define datasets
        if 'A_dev' in hf:
             ds_A = hf['A_dev']
             ds_W = hf['W_dev']
             ds_Xs = hf['X_s_dev']
             # ds_T removed
             if 'Y_dev' in hf:
                 ds_Y = hf['Y_dev']
             else:
                 raise KeyError("Y_dev dataset not found for RUL calculation")
                 
        elif 'Dev' in hf:
             ds_A = hf['Dev']['A']
             ds_W = hf['Dev']['W']
             ds_Xs = hf['Dev']['Xs']
             # ds_T removed
             if 'Y' in hf['Dev']:
                 ds_Y = hf['Dev']['Y']
             else:
                 raise KeyError("Y dataset not found in Dev group")
        else:
            raise KeyError("Dataset keys not found")
        
        total_rows = ds_A.shape[0]
        print(f"[DEV] Total rows in H5: {total_rows}")

        # --- PRE-CALCULATE CONTINUOUS RUL ---
        print("[DEV] Pre-calculating Continuous RUL...")
        # Load necessary columns for RUL calculation into memory
        # A: [unit, cycle, ...] -> Cols 0, 1
        # Y: [RUL] -> Col 0
        a_data = ds_A[:, :2] # Unit, Cycle
        y_data = ds_Y[:, 0]  # RUL
        
        # Create DataFrame for group operations
        df_rul = pd.DataFrame({'unit': a_data[:, 0], 'cycle': a_data[:, 1], 'RUL_raw': y_data})
        
        # Calculate steps within each cycle
        # We need cumulative count (current step, 1-based) and total count (total steps)
        # Assuming sorted by unit, cycle, but groupby is safer
        print("[DEV] Grouping by Unit/Cycle to calculate steps...")
        # Step: 1 to N
        df_rul['step'] = df_rul.groupby(['unit', 'cycle']).cumcount() + 1
        
        # Total steps per cycle
        # Transform returns the count aligned to original index
        df_rul['total_steps'] = df_rul.groupby(['unit', 'cycle'])['step'].transform('count')
        
        # Calculate Continuous RUL
        # Formula: RUL_discrete + 1 - (step / total_steps)
        # Example: RUL=60, Steps=2000. Step 1: 61 - 1/2000 = 60.9995. Step 2000: 61 - 1 = 60.0
        df_rul['RUL_continuous'] = df_rul['RUL_raw'] + 1.0 - (df_rul['step'] / df_rul['total_steps'])
        
        # Clip RUL at 73
        print("[DEV] Clipping RUL at 73...")
        df_rul['RUL_continuous'] = df_rul['RUL_continuous'].clip(upper=73.0)
        
        # Convert to float32 to match other data
        rul_array = df_rul['RUL_continuous'].values.astype('float32')
        
        # Clean up intermediate DF
        del df_rul, a_data, y_data
        gc.collect()

        # --- PREPARE PARQUET DATA ---
        print("[DEV] Loading Parquet data (columnar read to save memory)...")
        try:
            df_parquet = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error reading Parquet: {e}")
            return

        # drop_cols = ['unit', 'cycle', 'hs']
        # Remove 'hs' but KEEP 'unit' and 'cycle' for alignment check
        # We will drop them from chunk_pq later before concatenation if needed, 
        # OR we can keep them if we want to ensure they aren't duplicated in the final output
        # The prompt implies we are building a new schema, so let's check what we need.
        
        # Strategy: Keep 'unit' and 'cycle' in df_parquet for now to allow alignment check.
        # We will exclude them from the parquet part of the concatenation.
        
        if 'hs' in df_parquet.columns:
            df_parquet.drop(columns=['hs'], inplace=True)
        
        # Optimize types to float32
        float_cols = df_parquet.select_dtypes(include=['float64']).columns
        df_parquet[float_cols] = df_parquet[float_cols].astype('float32')
        
        print(f"[DEV] Parquet shape: {df_parquet.shape}")
        
        # --- ALIGNMENT & PADDING ---
        n_parquet = len(df_parquet)
        if n_parquet != total_rows:
            print(f"Row mismatch! H5: {total_rows}, Parquet: {n_parquet}")
            if n_parquet < total_rows:
                diff = total_rows - n_parquet
                print(f"Padding Parquet with {diff} rows by duplicating last row...")
                last_row = df_parquet.iloc[[-1]]
                padding = pd.concat([last_row] * diff, ignore_index=True)
                df_parquet = pd.concat([df_parquet, padding], ignore_index=True)
            elif n_parquet > total_rows:
                 print(f"Warning: Parquet has more rows. Truncating to {total_rows}...")
                 df_parquet = df_parquet.iloc[:total_rows] 
        
        df_parquet.reset_index(drop=True, inplace=True)
        
        # --- PREPARE COLUMNS FOR SCHEMA ---
        # 1. FC Columns
        # 1. FC Columns -> REMOVED (Single FC in dataset)
        # known_classes = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0] 
        # fc_cols = [f'FC_{int(val)}' for val in known_classes]
        fc_cols = []

        
        # 2. A columns
        a_cols = ['unit', 'cycle', 'hs']
        
        # 3. W columns
        w_cols = ['alt', 'Mach', 'TRA', 'T2']
        
        # 4. Xs columns
        xs_cols = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
        
        # 5. Parquet columns (Residuals)
        pq_cols = list(df_parquet.columns)
        
        # VALIDATION: Check for 14 residuals (New Architecture)
        if len(pq_cols) != 14:
            print(f"WARNING: Expected 14 residual columns (X only), found {len(pq_cols)}: {pq_cols}")
        else:
            print(f"[DEV] Confirmed 14 residual columns consistent with new architecture.")

        # 6. RUL column (New, replaces T_dev)
        rul_col = ['RUL']
        
        # New Order: a_cols -> fc_cols -> w_cols -> xs_cols -> pq_cols -> rul_col
        all_cols = a_cols + fc_cols + w_cols + xs_cols + pq_cols + rul_col
        print(f"[DEV] Total columns: {len(all_cols)}")
        print(f"[DEV] Columns: {all_cols}")
        
        # Write Loop
        writer = None
        processed_rows = 0
        
        while processed_rows < total_rows:
            end_row = min(processed_rows + chunk_size, total_rows)
            current_n = end_row - processed_rows
            
            # Read H5 chunks
            chunk_A = ds_A[processed_rows:end_row]
            chunk_W = ds_W[processed_rows:end_row]
            chunk_Xs = ds_Xs[processed_rows:end_row]
            # ds_T removed
            
            # Slice RUL array
            chunk_RUL = rul_array[processed_rows:end_row]

            # --- PROCESS FC ---
            # --- PROCESS FC ---
            # REMOVED: Dataset has only one FC value (3.0), encoding provides no info.
            
            # --- COMBINE ---
            chunk_pq = df_parquet.iloc[processed_rows:end_row].reset_index(drop=True)
            
            # --- STRICT ALIGNMENT CHECK ---
            # Check Unit Alignment
            h5_unit = chunk_A[:, 0]
            pq_unit = chunk_pq['unit'].values.astype('float32') # Ensure type match for comparison
            
            if not np.allclose(h5_unit, pq_unit, atol=1e-5):
                mismatch_idx = np.where(~np.isclose(h5_unit, pq_unit, atol=1e-5))[0]
                first_fail = mismatch_idx[0]
                raise ValueError(f"ALIGNMENT FAILURE at global row {processed_rows + first_fail}. "
                                 f"H5 Unit={h5_unit[first_fail]}, Parquet Unit={pq_unit[first_fail]}")
                                 
            # Check Cycle Alignment
            h5_cycle = chunk_A[:, 1]
            pq_cycle = chunk_pq['cycle'].values.astype('float32') # Ensure type match
            
            if not np.allclose(h5_cycle, pq_cycle, atol=1e-5):
                mismatch_idx = np.where(~np.isclose(h5_cycle, pq_cycle, atol=1e-5))[0]
                first_fail = mismatch_idx[0]
                raise ValueError(f"ALIGNMENT FAILURE at global row {processed_rows + first_fail}. "
                                 f"H5 Cycle={h5_cycle[first_fail]}, Parquet Cycle={pq_cycle[first_fail]}")
            
            # --- CLEANUP PARQUET CHUNK ---
            # Now that we've checked alignment, drop unit and cycle from chunk_pq
            # so they don't duplicate the ones we add from H5 (data_dict)
            cols_to_drop_pq = [c for c in ['unit', 'cycle'] if c in chunk_pq.columns]
            if cols_to_drop_pq:
                chunk_pq = chunk_pq.drop(columns=cols_to_drop_pq)

            data_dict = {}
            
            # A Cols
            if chunk_A.shape[1] >= 4:
                 data_dict['unit'] = chunk_A[:, 0]
                 data_dict['cycle'] = chunk_A[:, 1]
                 data_dict['hs'] = chunk_A[:, 3]
            else:
                 data_dict['unit'] = chunk_A[:, 0]
                 data_dict['cycle'] = chunk_A[:, 1]
                 data_dict['hs'] = chunk_A[:, 2]
            
            # FC Cols
            # FC Cols -> REMOVED
            # for idx, col in enumerate(fc_cols):
            #     data_dict[col] = fc_dummies[:, idx]
                
            # W Cols
            for idx, col in enumerate(w_cols):
                data_dict[col] = chunk_W[:, idx]
                
            # Xs Cols
            for idx, col in enumerate(xs_cols):
                data_dict[col] = chunk_Xs[:, idx]
                
            # Create base DF
            df_h5 = pd.DataFrame(data_dict)
            
            # Create RUL DF
            df_rul_chunk = pd.DataFrame({'RUL': chunk_RUL})

            # Concat: [H5 (FC, A, W, Xs)] + [Parquet] + [RUL]
            df_chunk = pd.concat([df_h5, chunk_pq, df_rul_chunk], axis=1)
            
            # Write
            table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            
            writer.write_table(table)
            
            processed_rows += current_n
            print(f"Processed {processed_rows}/{total_rows} rows ({(processed_rows/total_rows)*100:.1f}%)", flush=True)
            
            # Clean up
            del chunk_A, chunk_W, chunk_Xs, df_h5, df_chunk, table, chunk_pq, data_dict, df_rul_chunk, chunk_RUL
            gc.collect()

        if writer:
            writer.close()
        
        print("\n[DEV] Processing complete.")
        print("Final Columns:", all_cols)


def process_test_data(chunk_size=1000000):
    """
    Process Test set data (A_test, W_test, X_s_test) for RUL prediction.
    Logic from RUL/Mamba2_preprocess_test.py
    """
    # File paths
    h5_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\raw data\N-CMAPSS_DS02-006.h5'
    parquet_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\healthy state\inference_results_TEST.parquet'
    output_path = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\RUL\mamba2_processed_TEST.parquet'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[TEST] Processing TEST data in chunks of {chunk_size}...")

    # --- 0. PRE-VALIDATION ---
    print("[TEST] Checking file lengths...")
    with h5py.File(h5_path, 'r') as hf:
        if 'A_test' in hf:
            h5_rows = hf['A_test'].shape[0]
        elif 'Test' in hf:
            h5_rows = hf['Test']['A'].shape[0]
        else:
            raise KeyError("A_test or Test/A not found in H5")
            
    try:
        # Check if inference file exists (might still be generating)
        if not os.path.exists(parquet_path):
             raise FileNotFoundError(f"Inference file not found: {parquet_path}")

        # Read only unit column to get length quickly
        pq_meta = pd.read_parquet(parquet_path, columns=['unit']) 
        pq_rows = len(pq_meta)
        del pq_meta
    except Exception as e:
        raise RuntimeError(f"Could not read Parquet file: {e}")
        
    print(f"[TEST] H5 Rows: {h5_rows}, Parquet Rows: {pq_rows}")
    
    if h5_rows != pq_rows:
        diff = abs(h5_rows - pq_rows)
        if diff > 1000:
             raise ValueError(f"CRITICAL: Row count mismatch is too large ({diff}). H5={h5_rows}, Parquet={pq_rows}. Alignment is likely impossible.")
        else:
             print(f"WARNING: Row count mismatch ({diff}). Will attempt to pad/truncate, but alignment is not guaranteed.")
    else:
        print("[TEST] Row counts match perfectly.")

    # Open H5 for reading
    with h5py.File(h5_path, 'r') as hf:
        # Check keys and define datasets for TEST
        if 'A_test' in hf:
             ds_A = hf['A_test']
             ds_W = hf['W_test']
             ds_Xs = hf['X_s_test']
             if 'Y_test' in hf:
                 ds_Y = hf['Y_test']
             else:
                 raise KeyError("Y_test dataset not found for RUL calculation")
                 
        elif 'Test' in hf:
             ds_A = hf['Test']['A']
             ds_W = hf['Test']['W']
             ds_Xs = hf['Test']['Xs']
             if 'Y' in hf['Test']:
                 ds_Y = hf['Test']['Y']
             else:
                 raise KeyError("Y dataset not found in Test group")
        else:
            raise KeyError("Dataset keys not found")
        
        total_rows = ds_A.shape[0]
        print(f"[TEST] Total rows in H5 (Test): {total_rows}")

        # --- PRE-CALCULATE CONTINUOUS RUL ---
        print("[TEST] Pre-calculating Continuous RUL...")
        # Load necessary columns for RUL calculation into memory
        # A: [unit, cycle, ...] -> Cols 0, 1
        # Y: [RUL] -> Col 0
        a_data = ds_A[:, :2] # Unit, Cycle
        y_data = ds_Y[:, 0]  # RUL
        
        # Create DataFrame for group operations
        df_rul = pd.DataFrame({'unit': a_data[:, 0], 'cycle': a_data[:, 1], 'RUL_raw': y_data})
        
        # Calculate steps within each cycle
        print("[TEST] Grouping by Unit/Cycle to calculate steps...")
        # Step: 1 to N
        df_rul['step'] = df_rul.groupby(['unit', 'cycle']).cumcount() + 1
        
        # Total steps per cycle
        df_rul['total_steps'] = df_rul.groupby(['unit', 'cycle'])['step'].transform('count')
        
        # Calculate Continuous RUL (Consistent with Dev) -> REVERTED: User requested RAW RUL
        print("[TEST] Using RAWW RUL (clipped at 73) as requested...")
        # df_rul['RUL_continuous'] = df_rul['RUL_raw'] + 1.0 - (df_rul['step'] / df_rul['total_steps'])
        
        # Clip RUL at 73
        df_rul['RUL_clipped'] = df_rul['RUL_raw'].clip(upper=73.0)
        
        # Convert to float32
        rul_array = df_rul['RUL_clipped'].values.astype('float32')
        
        # Clean up intermediate DF
        del df_rul, a_data, y_data
        gc.collect()

        # --- PREPARE PARQUET DATA ---
        print("[TEST] Loading Parquet data (columnar read to save memory)...")
        try:
            df_parquet = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error reading Parquet: {e}")
            return

        if 'hs' in df_parquet.columns:
            df_parquet.drop(columns=['hs'], inplace=True)
        
        # Optimize types to float32
        float_cols = df_parquet.select_dtypes(include=['float64']).columns
        df_parquet[float_cols] = df_parquet[float_cols].astype('float32')
        
        print(f"[TEST] Parquet shape: {df_parquet.shape}")
        
        # --- ALIGNMENT & PADDING ---
        n_parquet = len(df_parquet)
        if n_parquet != total_rows:
            print(f"Row mismatch! H5: {total_rows}, Parquet: {n_parquet}")
            if n_parquet < total_rows:
                diff = total_rows - n_parquet
                print(f"Padding Parquet with {diff} rows by duplicating last row...")
                last_row = df_parquet.iloc[[-1]]
                padding = pd.concat([last_row] * diff, ignore_index=True)
                df_parquet = pd.concat([df_parquet, padding], ignore_index=True)
            elif n_parquet > total_rows:
                 print(f"Warning: Parquet has more rows. Truncating to {total_rows}...")
                 df_parquet = df_parquet.iloc[:total_rows] 
        
        df_parquet.reset_index(drop=True, inplace=True)
        
        # --- PREPARE COLUMNS FOR SCHEMA ---
        # 1. FC Columns
        # NEW: For Test data, we actually HAVE mixed FCs (1, 2, 3).
        # Should we encode them? 
        # A_test col 2 has FC.
        # Let's support standard FCs: 1, 2, 3.
        # But wait, did we train with FCs?
        # A_dev logic removed FCs because it was single value.
        # If we add FCs here, the schema differs from training data.
        # Ideally, we should match training schema.
        # So, NO FC columns for now to match training pipeline.
        fc_cols = []
        
        # 2. A columns
        a_cols = ['unit', 'cycle', 'hs']
        
        # 3. W columns
        w_cols = ['alt', 'Mach', 'TRA', 'T2']
        
        # 4. Xs columns
        # Ensure these match what's in H5 or what we want
        xs_cols = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
        
        # 5. Parquet columns
        pq_cols = list(df_parquet.columns)
        # remove unit/cycle from list if they are in df_parquet (we handle them)
        pq_cols = [c for c in pq_cols if c not in ['unit', 'cycle']]
        
        # VALIDATION: Check for 14 residuals
        if len(pq_cols) != 14:
             print(f"WARNING: Expected 14 residual columns (X only), found {len(pq_cols)}: {pq_cols}")
        else:
             print(f"[TEST] Confirmed 14 residual columns consistent with new architecture.")
        
        # 6. RUL column (New, replaces T_dev)
        rul_col = ['RUL']
        
        # New Order: a_cols -> fc_cols -> w_cols -> xs_cols -> pq_cols -> rul_col
        all_cols = a_cols + fc_cols + w_cols + xs_cols + pq_cols + rul_col
        print(f"[TEST] Total columns: {len(all_cols)}")
        print(f"[TEST] Columns: {all_cols}")
        
        # Write Loop
        writer = None
        processed_rows = 0
        
        while processed_rows < total_rows:
            end_row = min(processed_rows + chunk_size, total_rows)
            current_n = end_row - processed_rows
            
            # Read H5 chunks
            chunk_A = ds_A[processed_rows:end_row]
            chunk_W = ds_W[processed_rows:end_row]
            chunk_Xs = ds_Xs[processed_rows:end_row]
            
            # Slice RUL array
            chunk_RUL = rul_array[processed_rows:end_row]

            # --- PROCESS FC ---
            # REMOVED to match training schema
            
            # --- COMBINE ---
            chunk_pq = df_parquet.iloc[processed_rows:end_row].reset_index(drop=True)
            
            # --- STRICT ALIGNMENT CHECK ---
            # Check Unit Alignment
            h5_unit = chunk_A[:, 0]
            pq_unit = chunk_pq['unit'].values.astype('float32') # Ensure type match for comparison
            
            if not np.allclose(h5_unit, pq_unit, atol=1e-5):
                mismatch_idx = np.where(~np.isclose(h5_unit, pq_unit, atol=1e-5))[0]
                first_fail = mismatch_idx[0]
                raise ValueError(f"ALIGNMENT FAILURE at global row {processed_rows + first_fail}. "
                                 f"H5 Unit={h5_unit[first_fail]}, Parquet Unit={pq_unit[first_fail]}")
                                 
            # Check Cycle Alignment
            h5_cycle = chunk_A[:, 1]
            pq_cycle = chunk_pq['cycle'].values.astype('float32') # Ensure type match
            
            if not np.allclose(h5_cycle, pq_cycle, atol=1e-5):
                mismatch_idx = np.where(~np.isclose(h5_cycle, pq_cycle, atol=1e-5))[0]
                first_fail = mismatch_idx[0]
                raise ValueError(f"ALIGNMENT FAILURE at global row {processed_rows + first_fail}. "
                                 f"H5 Cycle={h5_cycle[first_fail]}, Parquet Cycle={pq_cycle[first_fail]}")
            
            # --- CLEANUP PARQUET CHUNK ---
            # Now that we've checked alignment, drop unit and cycle from chunk_pq
            # so they don't duplicate the ones we add from H5 (data_dict)
            cols_to_drop_pq = [c for c in ['unit', 'cycle'] if c in chunk_pq.columns]
            if cols_to_drop_pq:
                chunk_pq = chunk_pq.drop(columns=cols_to_drop_pq)

            data_dict = {}
            
            # A Cols
            if chunk_A.shape[1] >= 4:
                 data_dict['unit'] = chunk_A[:, 0]
                 data_dict['cycle'] = chunk_A[:, 1]
                 data_dict['hs'] = chunk_A[:, 3]
            else:
                 data_dict['unit'] = chunk_A[:, 0]
                 data_dict['cycle'] = chunk_A[:, 1]
                 data_dict['hs'] = chunk_A[:, 2] # Fallback if A is smaller
            
            # FC Cols -> Skipped
                
            # W Cols
            for idx, col in enumerate(w_cols):
                data_dict[col] = chunk_W[:, idx]
                
            # Xs Cols
            for idx, col in enumerate(xs_cols):
                data_dict[col] = chunk_Xs[:, idx]
                
            # Create base DF
            df_h5 = pd.DataFrame(data_dict)
            
            # Create RUL DF
            df_rul_chunk = pd.DataFrame({'RUL': chunk_RUL})

            # Concat: [H5 (FC, A, W, Xs)] + [Parquet] + [RUL]
            df_chunk = pd.concat([df_h5, chunk_pq, df_rul_chunk], axis=1)
            
            # Write
            table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            
            writer.write_table(table)
            
            processed_rows += current_n
            print(f"Processed {processed_rows}/{total_rows} rows ({(processed_rows/total_rows)*100:.1f}%)", flush=True)
            
            # Clean up
            del chunk_A, chunk_W, chunk_Xs, df_h5, df_chunk, table, chunk_pq, data_dict, df_rul_chunk, chunk_RUL
            gc.collect()

        if writer:
            writer.close()
        
        print("\n[TEST] Processing complete.")
        print("Final Columns:", all_cols)

def main():
    print("Starting combined RUL preprocessing pipeline execution...")
    
    # 1. Process Development data
    process_dev_data()
    
    # 2. Process Test data
    process_test_data()
    
    print("\nAll RUL preprocessing tasks completed successfully.")

if __name__ == "__main__":
    main()
