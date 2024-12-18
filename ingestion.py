import requests
import time
import yaml
from prefect import task, flow
import json
import pandas as pd
from datetime import datetime
import os


@task
def download_data(url: str, file_path: str):
    try:
        print(f"Starting download from {url}")
        response = requests.get(url)
        print(response.raise_for_status())  # Check for any errors in the request
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write content to file
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        print(f"Data successfully downloaded and saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download data: {e}")
        raise
    
    return file_path

@task
def make_df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df
    

@task
def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.rename(columns={
        "Aerosol_Optical_Depth_Land_Ocean_Mean_Min": "Aerosol_OD_Min",
        "Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": "Aerosol_OD_Mean",
        "Aerosol_Optical_Depth_Land_Ocean_Mean_Max": "Aerosol_OD_Max",
        "Aerosol_Optical_Depth_Land_Ocean_Mean_Std": "Aerosol_OD_Std"
    }, inplace=True)
    return data

@task
def save_renamed_data(data: pd.DataFrame, renamed_csv_path: str) -> str:
    data.to_csv(renamed_csv_path, index=False)  # Ensure not to save index
    return renamed_csv_path


@flow
def main_flow():
    url = "https://open.africa/dataset/9cef5972-8278-4669-962a-3d92f959d243/resource/9fd13a70-c726-4540-8293-4e66da1f51c6/download/africa_data_complete.csv"
    local_csv_path = "data/africa_dataset.csv"
    renamed_csv_path = "data/africa_dataset_renamed.csv"
    
    # Download the data
    file = download_data(url, local_csv_path)
    
    # Load the data into a DataFrame
    DataF = make_df(file)
    
    # Rename the columns
    renamed_df = rename_columns(DataF)
    
    # Save the renamed data
    saved_file_path = save_renamed_data(renamed_df, renamed_csv_path)
    
    print(f"Renamed data saved to {saved_file_path}")
    
    return saved_file_path
    
    


if __name__ == '__main__':
    main_flow()
