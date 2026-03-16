import os
import pandas as pd
import shutil

csv_path = r"C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\custom\divided_labels_consensus.csv"
df = pd.read_csv(csv_path)
df['Directory'] = df['Directory'].str.replace('\\', '/', regex=False)

df.to_csv("divided_labels_consensus_fs.csv", index=False)
