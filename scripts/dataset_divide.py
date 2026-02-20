import os
import pandas as pd
import shutil

def organize_files(source_directory, master_directory, subfolder_name, dataset_csv,files_per_folder=9000):    
    #Check if the subfolder directory exists, if not create it
    dataset_dir = os.path.join(master_directory, subfolder_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print("Subfolder ready.")
    
    current_partition = 0
    df = pd.read_csv(dataset_csv)

    #Insert empty directory column in the dataframe
    df.insert(0, 'Directory', None)

    print("Starting file organization...")
    for i in range(0, len(df), files_per_folder):
        current_partition += 1
        partition_path = os.path.join(dataset_dir, f"part_{current_partition}")
        os.makedirs(partition_path, exist_ok=True)

        partition_files = df.iloc[i:i+files_per_folder]
        print(f"Processing partition {current_partition} with {len(partition_files)} files...")

        #add directory to directory column in the dataframe
        location = os.path.relpath(partition_path, start=master_directory)
        df.loc[partition_files.index, 'Directory'] = location

        for _, row in partition_files.iterrows():
            file_name = row['FileName']
            source_path = os.path.join(source_directory, file_name)
            destination_path = os.path.join(partition_path, file_name)

            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
            else:
                print(f"Warning: Source file not found: {file_name}")

    # Save the updated dataframe back to CSV
    df.to_csv("output/labels_wdir_csv", index=False)
    print("Organization complete.")

source_dir = r"C:\Datasets\MSP-PODCAST-Publish-2.0\Audios"
master_dir = r"C:\Datasets\_compiled"
dataset_name = "msp-podcast-2_divided"
csv = r"C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\labels_consensus.csv"

organize_files(source_dir, master_dir, dataset_name, csv)