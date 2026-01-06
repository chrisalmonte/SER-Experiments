import os
import shutil

def organize_files(source_directory, destination_directory, partitions_file_path):    
    # Check if the partitions file exists
    if not os.path.exists(partitions_file_path):
        print(f"Error: The file '{partitions_file_path}' was not found.")
        return

    print("Starting file organization...")

    with open(partitions_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            
            # The format is "Label; Filename.wav"
            parts = line.strip().split(';')
            
            if len(parts) >= 2:
                label_folder = parts[0].strip()
                file_name = parts[1].strip()
                
                # Construct the full path to the source file
                source_path = os.path.join(source_directory, file_name)
                
                # Construct the destination folder path
                destination_folder = os.path.join(destination_directory, label_folder)
                destination_path = os.path.join(destination_folder, file_name)

                # Copy the file if it exists in the source directory
                if os.path.exists(source_path):
                    shutil.copy2(source_path, destination_path)
                    # print(f"Copied: {file_name} -> {label_folder}")
                else:
                    print(f"Warning: Source file not found: {file_name}")
    print("Organization complete.")

source_folder = r"C:\Datasets\MSP-PODCAST-Publish-2.0\Audios"
partitions_file = r"C:\Datasets\MSP-PODCAST-Publish-2.0\Partitions.txt"
destination_folder = r"C:\Datasets\_compiled\msp-podcast-2"

organize_files(source_folder, destination_folder,partitions_file)
