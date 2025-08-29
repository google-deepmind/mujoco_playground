'''
A script to change the motor data to the tendon length data based on the following transform:

Servo Diff   Angle (deg)     Angle (rad)     tendon length (m)   
---------------------------------------------
0     1220   107.227         1.871           0.01684        
1     1200   105.469         1.841           0.01657        
2     1495   131.396         2.293           0.02064        
5     3220   283.008         4.939           0.04445        
8     3220   283.008         4.939           0.04445        
11    3220   283.008         4.939           0.04445        
14    3220   283.008         4.939           0.04445


 Right Hand (indices 0-14):

  Thumb:
  - Index 0: Right thumb CMC abduction
  - Index 1: Right thumb CMC flexion
  - Index 2: Right thumb tendon

  Index Finger:
  - Index 3: Right index MCP (prismatic 1)
  - Index 4: Right index PIP (prismatic 2)
  - Index 5: Right index tendon

  Middle Finger:
  - Index 6: Right middle MCP (prismatic 1)
  - Index 7: Right middle PIP (prismatic 2)
  - Index 8: Right middle tendon

  Ring Finger:
  - Index 9: Right ring MCP (prismatic 1)
  - Index 10: Right ring PIP (prismatic 2)
  - Index 11: Right ring tendon

  Pinky:
  - Index 12: Right pinky MCP (prismatic 1)
  - Index 13: Right pinky PIP (prismatic 2)
  - Index 14: Right pinky tendon


After the data is transformed to tendon length, it is then imported to the 
step_response_tendon_from_real_control.py to test the sim-real gap
'''

import os
import re
import glob
import pandas as pd
import numpy as np

# Data folder path
DATA_FOLDER = "./data"

# Tendon indices mapping (based on the comment above)
TENDON_INDICES = [1, 2, 5, 8, 11, 14]  # These correspond to the tendon actuators
MAX_ENCODER = 65535

# Servo difference to tendon length mapping (from the table above)
SERVO_TO_TENDON_MAPPING = {
    0: 0.01684/MAX_ENCODER,
    1: 0.01657/MAX_ENCODER,
    2: 0.02064/MAX_ENCODER,
    5: 0.04445/MAX_ENCODER, 
    8: 0.04445/MAX_ENCODER,  
    11: 0.04445/MAX_ENCODER, 
    14: 0.04445/MAX_ENCODER
}

def extract_number_from_filename(filename):
    """
    Extract the number from filename (e.g., 'motor_14.csv' -> 14)
    """
    # Look for numbers in the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])  # Return the first number found
    return None

def find_csv_files_with_numbers():
    """
    Find all CSV files in the data folder and extract their numbers
    """
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    files_with_numbers = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Skip the output file if it already exists
        if filename == "data.csv":
            continue
        number = extract_number_from_filename(filename)
        if number is not None:
            files_with_numbers.append((csv_file, number))
    
    return files_with_numbers

def load_motor_data(csv_path, tendon_idx):
    """
    Load motor data from CSV file and extract servo_{tendon_idx}_sent and servo_{tendon_idx}_read columns
    """
    print(f"Loading motor data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check if the required columns exist
        sent_column = f'servo_{tendon_idx}_sent'
        read_column = f'servo_{tendon_idx}_read'
        
        if sent_column not in df.columns or read_column not in df.columns:
            print(f"Warning: Missing columns {sent_column} or {read_column}")
            print(f"Available columns: {list(df.columns)}")
            return None, None, df
        
        # Extract the servo data
        servo_sent = df[sent_column].values
        servo_read = df[read_column].values
        
        print(f"Loaded {len(servo_sent)} data points")
        print(f"Servo sent range: [{servo_sent.min():.2f}, {servo_sent.max():.2f}]")
        print(f"Servo read range: [{servo_read.min():.2f}, {servo_read.max():.2f}]")
        
        return servo_sent, servo_read, df
        
    except Exception as e:
        print(f"Error loading file {csv_path}: {e}")
        return None, None, None

def transform_servo_to_tendon(servo_sent, servo_read, tendon_idx):
    """
    Transform servo data to tendon length data based on the mapping
    """
    print(f"Transforming servo data to tendon lengths for tendon {tendon_idx}...")
    
    if tendon_idx not in SERVO_TO_TENDON_MAPPING:
        print(f"Warning: No mapping found for tendon {tendon_idx}")
        return None, None
    
    # Get the scaling factor for this tendon
    scaling_factor = SERVO_TO_TENDON_MAPPING[tendon_idx]
    
    # Transform servo values to tendon lengths
    tendon_sent = servo_sent * scaling_factor
    tendon_read = servo_read * scaling_factor
    
    print(f"Transformed data shape: {tendon_sent.shape}")
    print(f"Tendon sent range: [{tendon_sent.min():.6f}, {tendon_sent.max():.6f}]")
    print(f"Tendon read range: [{tendon_read.min():.6f}, {tendon_read.max():.6f}]")
    
    return tendon_sent, tendon_read

def process_single_file(csv_path, tendon_idx):
    """
    Process a single CSV file for a specific tendon and return transformed data
    """
    print(f"\n=== Processing file: {os.path.basename(csv_path)} for tendon {tendon_idx} ===")
    
    # Load motor data
    servo_sent, servo_read, df = load_motor_data(csv_path, tendon_idx)
    if servo_sent is None:
        print(f"Skipping file due to missing data")
        return None
    
    # Transform servo data to tendon lengths
    tendon_sent, tendon_read = transform_servo_to_tendon(servo_sent, servo_read, tendon_idx)
    if tendon_sent is None:
        print(f"Skipping file due to transformation error")
        return None
    
    # Create a new dataframe with tendon data
    tendon_df = pd.DataFrame({
        f'tendon_{tendon_idx}_sent': tendon_sent,
        f'tendon_{tendon_idx}_read': tendon_read
    })
    
    return tendon_df

def save_all_data_to_csv(all_data, output_path):
    """
    Save all transformed data to a single CSV file
    """
    print(f"Saving all transformed data to: {output_path}")
    
    # Combine all dataframes
    if all_data:
        combined_df = pd.concat(all_data, axis=1)
        combined_df.to_csv(output_path, index=False)
        print(f"Data saved successfully to {output_path}!")
        print(f"Final data shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        return combined_df
    else:
        print("No data to save!")
        return None

def main():
    print("=== Motor Data to Tendon Length Transformer ===")
    
    # Check if data folder exists
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder not found: {DATA_FOLDER}")
        return
    
    # Find all CSV files with numbers
    files_with_numbers = find_csv_files_with_numbers()
    
    if not files_with_numbers:
        print(f"No CSV files with numbers found in {DATA_FOLDER}")
        return
    
    print(f"Found {len(files_with_numbers)} CSV files to process:")
    for csv_path, number in files_with_numbers:
        print(f"  - {os.path.basename(csv_path)} (number: {number})")
    
    # Process each file and collect all data
    all_data = []
    processed_files = []
    
    for csv_path, number in files_with_numbers:
        # Check if this number corresponds to a tendon index
        if number in TENDON_INDICES:
            tendon_df = process_single_file(csv_path, number)
            if tendon_df is not None:
                all_data.append(tendon_df)
                processed_files.append((csv_path, number))
        else:
            print(f"Skipping {os.path.basename(csv_path)}: number {number} not in tendon indices {TENDON_INDICES}")
    
    # Save all data to single CSV file
    output_path = os.path.join(DATA_FOLDER, "real_control.csv")
    combined_df = save_all_data_to_csv(all_data, output_path)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(processed_files)} files:")
    
    if combined_df is not None:
        print(f"\nFinal output file: {output_path}")
        print(f"Contains tendon data for: {[f'tendon_{num}' for _, num in processed_files]}")

if __name__ == "__main__":
    main()