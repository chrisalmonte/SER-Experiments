import numpy as np
import os
import pandas as pd
from cAudiotools import Utils
from funasr import AutoModel
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":

    model_id = "emotion2vec/emotion2vec_base"
    model = AutoModel(
        model=model_id,
        hub="hf", 
    )

    labels = pd.read_csv(r"C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\custom\divided_labels_consensus.csv")
    master_dir = r"C:\Datasets\_compiled"
    labels['FullPath'] = labels.iloc[:, 0].str.cat(labels.iloc[:, 1], sep='\\', na_rep='')
    labels['FullPath'] = labels.iloc[:, 0].str.cat(labels.iloc[:, 1], sep='\\', na_rep='')
    audio_paths = labels['FullPath'].tolist()
    
    for path in tqdm(audio_paths):
        path = os.path.join(master_dir, path)
        file_name = Path(path).stem
        audio = Utils.load_as_np(path)
        rec_result = model.generate(audio[0], output_dir=None, granularity="utterance")
        embeddings = np.array(rec_result[0]['feats'])
        np.save(f"output/processing/e2v_embeddings/{file_name}.npy", embeddings)