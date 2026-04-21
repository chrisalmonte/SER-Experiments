import os
import pandas as pd

CREMAD_ROOT = r"C:\Datasets\crema-d-mirror\AudioWAV"
OUTPUT_CSV_FILE = "cremad_labels.csv"
DEMOGRAPHICS_FILE = r"C:\Datasets\crema-d-mirror\VideoDemographics.csv"

# Mapping following MSP-Podcast nomenclature
EMOTION_MAP = {
    'NEU': 'N', # Neutral
    'HAP': 'H', # Happiness
    'SAD': 'S', # Sadness
    'ANG': 'A', # Anger
    'FEA': 'F', # Fear
    'DIS': 'D', # Disgust
}

INTENSITY_MAP = {
    'LO': 'low',
    'MD': 'medium',
    'HI': 'high',
    'XX': 'unspecified'
}

def parse_filename(filename, demography_df):
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) != 4:
        print("Invalid filename format:", filename)
        return None
    
    subdir = "crema-d/AudioWAV"
    actor_code = int(parts[0])
    statement_code = parts[1]
    emotion_code = parts[2]
    intensity_code = parts[3]
    
    gender_series = demography_df.loc[demography_df['ActorID'] == actor_code, 'Sex']
    gender = gender_series.values[0] if not gender_series.empty else 'Unknown'
    speaker_id = f"RVS_{actor_code}"
    
    return {
        'Directory': subdir,
        'FileName': filename,
        'EmoClass': EMOTION_MAP.get(emotion_code, 'N/A'),
        'EmoInt': INTENSITY_MAP.get(intensity_code, 'N/A'),
        'Statement': statement_code, 
        'SpkrID': speaker_id,              
        'Gender': gender
    }

dataset_rows = []
demography_df = pd.read_csv(DEMOGRAPHICS_FILE)

for filename in os.listdir(CREMAD_ROOT):            
    row_data = parse_filename(filename, demography_df)
    if row_data:
        dataset_rows.append(row_data)

if dataset_rows:
    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Saved to: {OUTPUT_CSV_FILE}\n")
    print(df.head(10))
