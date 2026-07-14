from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
import audioflux
import pandas as pd
import os
import tqdm

model = load_silero_vad()
files = []

def get_timestamps(audio_path):
    audio, sample_rate = audioflux.read(path=audio_path)
    if sample_rate != 16000:
        audio = audioflux.resample(audio, sample_rate, 16000)
        sample_rate = 16000

    speech_timestamps = get_speech_timestamps(audio, model, return_seconds=False)
    return speech_timestamps

#load dataframe from csv
df = pd.read_csv("output/processing/custom_labels/crema-d/cremad_labels.csv")
root = r"C:\Datasets\_compiled"

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    audio_path = os.path.join(root, row['Directory'], row['FileName'])
    speech_timestamps = get_timestamps(audio_path)
    if len(speech_timestamps) > 1:
        files.append(row['FileName'])

for file in files:
    print(file)