import pandas as pd
import os

def save_data_to_csv(directory_path: str, file_name: str, data: dict):
    """
    Saves a dictionary to a CSV file. If the file exists, it appends the data
    while remaining tolerant of different column sets.

    Args:
        directory_path (str): The system path where the file should be saved.
        file_name (str): The name of the CSV file (including .csv extension).
        data (dict): A dictionary representing the data to save, where keys 
            are column names and values are the data points.

    Returns:
        None: The function performs an in-place file operation.
    """
    # Combine path and filename
    full_path = os.path.join(directory_path, file_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Convert the input dictionary to a DataFrame
    # Using [data] creates a single-row DataFrame
    new_df = pd.DataFrame([data])

    if os.path.exists(full_path):
        # Read the existing CSV
        existing_df = pd.read_csv(full_path)
        
        # Concatenate existing and new data
        # 'sort=False' preserves column order; missing columns are filled with NaN
        updated_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        
        # Save back to CSV
        updated_df.to_csv(full_path, index=False)
    else:
        # File doesn't exist, just create it
        new_df.to_csv(full_path, index=False)

# --- Example Usage ---
# data_sample = {"name": "Alice", "age": 30, "city": "Madrid"}
# save_data_to_csv("./output", "records.csv", data_sample)