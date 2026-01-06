import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset

class AudioDatasetCategory(Dataset):
    def __init__(self, annotations_file, audio_dir, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        audio, sample_rate = torchaudio.load(audio_path)
        label = self.audio_labels.iloc[idx, 1]
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        return audio, label

class AudioDatasetVAD(Dataset):
    def __init__(self, annotations_file, audio_dir, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        audio, sample_rate = torchaudio.load(audio_path)
        val = self.audio_labels.iloc[idx, 2]
        act = self.audio_labels.iloc[idx, 1]
        dom = self.audio_labels.iloc[idx, 3]
        vad = torch.tensor([val, act, dom], dtype=torch.float32)
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            vad = self.target_transform(vad)
        return audio, vad
    