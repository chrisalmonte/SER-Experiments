import os
import pandas as pd

DATASET_DIRECTORY = "/home/imd-temp/datasets"
RAVDESS_ROOT = "ravdess/Audio_Speech_Actors_01-24" 
OUTPUT_CSV_FILE = "ravdess_labels_speech.csv"

# Mapping following MSP-Podcast nomenclature
# Calm will be labeled as neutral.
EMOTION_MAP = {
    '01': 'N', # Neutral
    '02': 'N', # Calm
    '03': 'H', # Happiness
    '04': 'S', # Sadness
    '05': 'A', # Anger
    '06': 'F', # Fear
    '07': 'D', # Disgust
    '08': 'U'  # Surprise
}

INTENSITY_MAP = {
    '01': 'normal',
    '02': 'strong'
}

def parse_ravdess_filename(filename, subdir):
    """Parses a RAVDESS filename and returns a dictionary of attributes."""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('-')
    
    if len(parts) != 7:
        return None

    emotion_code = parts[2]
    intensity_code = parts[3]
    statement_code = parts[4]
    actor_code = parts[6]
    
    speaker_id = int(actor_code)
    statement_code = int(statement_code)
    gender = 'Female' if speaker_id % 2 == 0 else 'Male'
    
    return {
        'Directory': subdir,
        'FileName': filename,
        'EmoClass': EMOTION_MAP.get(emotion_code, 'unknown'),
        'EmoInt': INTENSITY_MAP.get(intensity_code, 'unknown'),
        'Statement': statement_code, 
        'SpkrID': speaker_id,              
        'Gender': gender
    }

dataset_rows = []

for actor in range(1, 25):
    subdir = f"{RAVDESS_ROOT}/Actor_{actor:02d}"
    actor_dir = f"{DATASET_DIRECTORY}/{subdir}"
    if not os.path.exists(actor_dir):
        print(f"Warning: Actor directory not found: {actor_dir}")
        continue

    print(f"Scanning directory: {actor_dir}...")

    for filename in os.listdir(actor_dir):            
        row_data = parse_ravdess_filename(filename, subdir)
        if row_data:
            dataset_rows.append(row_data)

if dataset_rows:
    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Saved to: {OUTPUT_CSV_FILE}\n")
    print(df.head(10))
