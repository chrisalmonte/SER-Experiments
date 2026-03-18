import numpy as np
import os
import pandas as pd
from cAudiotools import Utils
from funasr import AutoModel
from pathlib import Path

if __name__ == "__main__":

    model_id = "emotion2vec/emotion2vec_base"
    granularity = "frame"
    model = AutoModel(
        model=model_id,
        hub="hf", 
    )

    master_dir = "/home/imd-temp/datasets/"
    labels = pd.read_csv("/home/imd-temp/datasets/ravdess/labels/ravdess_labels_speech.csv")
    labels['FullPath'] = labels.iloc[:, 0].str.cat(labels.iloc[:, 1], sep='/', na_rep='')
    audio_paths = labels['FullPath'].tolist()
    
    for path in audio_paths:
        path = os.path.join(master_dir, path)
        file_name = Path(path).stem
        audio = Utils.load_as_np(path)
        rec_result = model.generate(audio[0], output_dir=None, granularity=granularity)
        embeddings = np.array(rec_result[0]['feats'])
        np.save(f"output/processing/e2v_embeddings_{granularity}/{file_name}.npy", embeddings)