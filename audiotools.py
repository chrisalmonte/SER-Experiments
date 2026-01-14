import matplotlib.pyplot as plt
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
    
class Plot:
    @staticmethod
    def waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None, size=(12, 4)):        
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        plt.figure(figsize=size)
        plt.plot(time_axis, waveform[0])
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def spectrogram(spectrogram, title="Spectrogram", ylabel="Frequency bin", xlabel="Frame", size=(12, 4)):
        spectrogram = spectrogram.squeeze().numpy()
        
        # Convert to dB scale
        spectrogram_db = 10 * torch.log10(torch.tensor(spectrogram) + 1e-10).numpy()

        plt.figure(figsize=size)
        plt.imshow(spectrogram_db, origin='lower', aspect='auto', cmap='magma')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.colorbar(format='%+2.0f dB')
        plt.show()

class Transforms:
    @staticmethod
    def mono(waveform):
        if waveform.size(0) > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    @staticmethod
    def resample(waveform, orig_freq, new_freq):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(waveform)

    @staticmethod
    def pad_trim(waveform, max_len):
        if waveform.size(1) > max_len:
            return waveform[:, :max_len]
        else:
            padding = max_len - waveform.size(1)
            return torch.nn.functional.pad(waveform, (0, padding))

class Batching:
    @staticmethod
    def waveform_dynamic(batch):
        max_len = max(item[0].size(1) for item in batch)
        batch_inputs = []
        batch_targets = [item[1] for item in batch]
        for item in batch:
            batch_inputs.append(Transforms.pad_trim(item[0], max_len))
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets)
        return batch_inputs, batch_targets
    
    @staticmethod
    def spectrogram_dynamic(batch):
        max_freq_bins = max(item[0].size(1) for item in batch)
        max_time_frames = max(item[0].size(2) for item in batch)
        batch_inputs = []
        batch_targets = [item[1] for item in batch]
        for item in batch:
            raw_spectogram = item[0]
            padded_spec = torch.nn.functional.pad(raw_spectogram, (0, max_time_frames - raw_spectogram.size(2), 0, max_freq_bins - raw_spectogram.size(1)))
            batch_inputs.append(padded_spec)
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets)
        return batch_inputs, batch_targets

        
    