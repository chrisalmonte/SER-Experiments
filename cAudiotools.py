import audioflux
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
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
        audio = Utils.load_4_torch(audio_path)
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
        audio = Utils.load_4_torch(audio_path)
        val = self.audio_labels.iloc[idx, 2]
        act = self.audio_labels.iloc[idx, 1]
        dom = self.audio_labels.iloc[idx, 3]
        vad = torch.tensor([val, act, dom], dtype=torch.float32)
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            vad = self.target_transform(vad)
        return audio, vad

class VADSubdirAudioDataset(Dataset):
    def __init__(self, annotations_file, master_dir, vad_column_names, transform=None, target_transform=None, 
                 subdir_column_name=None, name_column_name=None, include_only: tuple=None):
        self.labels = pd.read_csv(annotations_file)
        self.val_idx = self.labels.columns.get_loc(vad_column_names[0])
        self.act_idx = self.labels.columns.get_loc(vad_column_names[1])
        self.dom_idx = self.labels.columns.get_loc(vad_column_names[2])
        self.master_dir = master_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subdir_idx = 0 if subdir_column_name is None else self.labels.columns.get_loc(subdir_column_name)
        self.name_idx = 1 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only is not None:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.master_dir, self.labels.iloc[idx, self.subdir_idx], self.labels.iloc[idx, self.name_idx])
        audio, sample_rate = Utils.load_as_np(audio_path)
        val = self.labels.iloc[idx, self.val_idx]
        act = self.labels.iloc[idx, self.act_idx]
        dom = self.labels.iloc[idx, self.dom_idx]
        vad = torch.tensor([val, act, dom], dtype=torch.float32)
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            vad = self.target_transform(vad)
        audio = torch.from_numpy(audio).float()
        return audio, vad
    
class VADSubdirAudioDatasetGender(Dataset):
    def __init__(self, annotations_file, master_dir, vad_column_names, gender_column_name, transform=None, target_transform=None, 
                 subdir_column_name=None, name_column_name=None, include_only: tuple=None, gender_map_dict=None):
        self.labels = pd.read_csv(annotations_file)
        self.val_idx = self.labels.columns.get_loc(vad_column_names[0])
        self.act_idx = self.labels.columns.get_loc(vad_column_names[1])
        self.dom_idx = self.labels.columns.get_loc(vad_column_names[2])
        self.gender_idx = self.labels.columns.get_loc(gender_column_name)
        self.master_dir = master_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subdir_idx = 0 if subdir_column_name is None else self.labels.columns.get_loc(subdir_column_name)
        self.name_idx = 1 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only is not None:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)
        
        if gender_map_dict:
            col_name = self.labels.columns[self.gender_idx]
            self.labels[col_name] = self.labels[col_name].map(gender_map_dict)
            col_name = self.labels.columns[self.gender_idx]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.master_dir, self.labels.iloc[idx, self.subdir_idx], self.labels.iloc[idx, self.name_idx])
        audio, sample_rate = Utils.load_as_np(audio_path)
        val = self.labels.iloc[idx, self.val_idx]
        act = self.labels.iloc[idx, self.act_idx]
        dom = self.labels.iloc[idx, self.dom_idx]
        vad = torch.tensor([val, act, dom], dtype=torch.float32)
        gender = self.labels.iloc[idx, self.gender_idx]
        gender = torch.tensor(gender, dtype=torch.long)
        gender = torch.nn.functional.one_hot(gender, num_classes=3).float()
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            vad = self.target_transform(vad)
        audio = torch.from_numpy(audio).float()
        return audio, vad, gender    

class ClassSubdirAudioDataset(Dataset):
    def __init__(self, annotations_file, master_dir, class_column_name, transform=None, target_transform=None, 
                 subdir_column_name=None, name_column_name=None, include_only: tuple=None, map_dict=None):
        self.labels = pd.read_csv(annotations_file)
        self.class_idx = self.labels.columns.get_loc(class_column_name)
        self.master_dir = master_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subdir_idx = 0 if subdir_column_name is None else self.labels.columns.get_loc(subdir_column_name)
        self.name_idx = 1 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)

        if map_dict:
            col_name = self.labels.columns[self.class_idx]
            self.labels[col_name] = self.labels[col_name].map(map_dict)
            col_name = self.labels.columns[self.class_idx]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.master_dir, self.labels.iloc[idx, self.subdir_idx], self.labels.iloc[idx, self.name_idx])
        audio, sample_rate = Utils.load_as_np(audio_path)
        class_label = self.labels.iloc[idx, self.class_idx]
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            class_label = self.target_transform(class_label)
        audio = torch.from_numpy(audio).float()
        class_label = torch.tensor(class_label, dtype=torch.long)
        return audio, class_label
    
class ClassDFSubdirAudioDataset(Dataset):
    def __init__(self, labels_df, master_dir, class_column_name, transform=None, target_transform=None, 
                 subdir_column_name=None, name_column_name=None, include_only: tuple=None, map_dict=None):
        self.labels = labels_df
        self.class_idx = self.labels.columns.get_loc(class_column_name)
        self.master_dir = master_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subdir_idx = 0 if subdir_column_name is None else self.labels.columns.get_loc(subdir_column_name)
        self.name_idx = 1 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)

        if map_dict:
            col_name = self.labels.columns[self.class_idx]
            self.labels[col_name] = self.labels[col_name].map(map_dict)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.master_dir, self.labels.iloc[idx, self.subdir_idx], self.labels.iloc[idx, self.name_idx])
        audio, sample_rate = Utils.load_as_np(audio_path)
        class_label = self.labels.iloc[idx, self.class_idx]
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            class_label = self.target_transform(class_label)
        audio = torch.from_numpy(audio).float()
        class_label = torch.tensor(class_label, dtype=torch.long)
        return audio, class_label
    
class ClassSubdirAudioDatasetRS(Dataset):
    def __init__(self, annotations_file, master_dir, class_column_name, transform=None, target_transform=None, 
                 subdir_column_name=None, name_column_name=None, include_only: tuple=None, map_dict=None, target_sample_rate=16000):
        self.labels = pd.read_csv(annotations_file)
        self.class_idx = self.labels.columns.get_loc(class_column_name)
        self.master_dir = master_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subdir_idx = 0 if subdir_column_name is None else self.labels.columns.get_loc(subdir_column_name)
        self.name_idx = 1 if name_column_name is None else self.labels.columns.get_loc(name_column_name)
        self.target_sample_rate = target_sample_rate

        if include_only:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)

        if map_dict:
            col_name = self.labels.columns[self.class_idx]
            self.labels[col_name] = self.labels[col_name].map(map_dict)
            col_name = self.labels.columns[self.class_idx]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.master_dir, self.labels.iloc[idx, self.subdir_idx], self.labels.iloc[idx, self.name_idx])
        audio, sample_rate = Utils.load_resample_as_np(audio_path, self.target_sample_rate)
        class_label = self.labels.iloc[idx, self.class_idx]
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            class_label = self.target_transform(class_label)
        audio = torch.from_numpy(audio).float()
        class_label = torch.tensor(class_label, dtype=torch.long)
        return audio, class_label

class VADEmbeddingsDataset(Dataset):
    def __init__(self, annotations_file, embeddings_dir, vad_column_names, transform=None, target_transform=None, 
                 name_column_name=None, include_only: tuple=None):
        self.labels = pd.read_csv(annotations_file)
        self.val_idx = self.labels.columns.get_loc(vad_column_names[0])
        self.act_idx = self.labels.columns.get_loc(vad_column_names[1])
        self.dom_idx = self.labels.columns.get_loc(vad_column_names[2])
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        self.target_transform = target_transform
        self.name_idx = 0 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embeddings_dir, self.labels.iloc[idx, self.name_idx].replace('.wav', '.npy'))
        embedding = torch.from_numpy(np.load(embedding_path)).float()
        val = self.labels.iloc[idx, self.val_idx]
        act = self.labels.iloc[idx, self.act_idx]
        dom = self.labels.iloc[idx, self.dom_idx]
        vad = torch.tensor([val, act, dom], dtype=torch.float32)
        if self.transform:
            embedding = self.transform(embedding)
        if self.target_transform:
            vad = self.target_transform(vad)
        return embedding, vad
    
class ClassEmbeddingsDataset(Dataset):
    def __init__(self, annotations_file, embeddings_dir, class_column_name, mappings_dict=None, transform=None, target_transform=None, 
                 name_column_name=None, include_only: tuple=None):
        self.labels = pd.read_csv(annotations_file)
        self.class_idx = self.labels.columns.get_loc(class_column_name)
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        self.target_transform = target_transform
        self.name_idx = 0 if name_column_name is None else self.labels.columns.get_loc(name_column_name)

        if include_only:
            self.labels = self.labels[self.labels[include_only[0]].isin(include_only[1])].reset_index(drop=True)
        
        if mappings_dict:
            col_name = self.labels.columns[self.class_idx]
            self.labels[col_name] = self.labels[col_name].map(mappings_dict)
            col_name = self.labels.columns[self.class_idx]
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embeddings_dir, self.labels.iloc[idx, self.name_idx].replace('.wav', '.npy'))
        embedding = torch.from_numpy(np.load(embedding_path)).float()
        class_label = self.labels.iloc[idx, self.class_idx]
        if self.transform:
            embedding = self.transform(embedding)
        if self.target_transform:
            class_label = self.target_transform(class_label)
        return embedding, class_label

class Collate:    
    @staticmethod
    def waveform_dynamic(batch):
        max_len = max(item[0].size(1) for item in batch)
        batch_inputs = []
        batch_targets = [item[1] for item in batch]
        for item in batch:
            padded_input = F.pad(item[0], (0, max_len - item[0].size(1)))
            batch_inputs.append(padded_input)
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
            padded_spec = F.pad(raw_spectogram, (0, max_time_frames - raw_spectogram.size(2), 0, max_freq_bins - raw_spectogram.size(1)))
            batch_inputs.append(padded_spec)
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets)
        return batch_inputs, batch_targets
    
    @staticmethod
    def waveform_dynamic_wLengths(batch):
        audio_waveforms, batch_targets = zip(*batch)
        batch_inputs = []
        waveform_lengths = torch.tensor(
            [waveform.shape[-1] for waveform in audio_waveforms],
            dtype=torch.long
        )
        max_len = max(waveform_lengths).max().item()
        for waveform in audio_waveforms:
            batch_inputs.append(F.pad(waveform, (0, max_len - waveform.shape[-1])))
        batch_inputs = torch.stack(batch_inputs, dim=0)
        batch_targets = torch.stack(batch_targets, dim=0)
        return batch_inputs, waveform_lengths, batch_targets

    @staticmethod
    def waveform_dynamic_wMasks(batch):
        audio_waveforms, batch_targets = zip(*batch)

        # pad_sequence expects a list of 1D tensors (Time,), not (1, Time)
        processed_waveforms = []
        lengths = []
        for wav in audio_waveforms:
            squeezed_wav = wav.squeeze()
            processed_waveforms.append(squeezed_wav)
            lengths.append(squeezed_wav.size(0))

        batch_inputs = torch.nn.utils.rnn.pad_sequence(processed_waveforms, batch_first=True, padding_value=0.0)

        #Create Masks (Vectorized)
        max_len = batch_inputs.shape[1]

        lengths_tensor = torch.tensor(lengths).unsqueeze(1) # Shape: (Batch, 1)
        range_tensor = torch.arange(max_len).unsqueeze(0)   # Shape: (1, Max_Len)
        batch_masks = (range_tensor < lengths_tensor).long()

        #Stack Targets
        batch_targets = torch.stack(batch_targets, dim=0)

        return batch_inputs, batch_masks, batch_targets

    @staticmethod
    def waveform_dynamic_wMasks_gender(batch):
        audio_waveforms, batch_targets, genders = zip(*batch)

        # pad_sequence expects a list of 1D tensors (Time,), not (1, Time)
        processed_waveforms = []
        lengths = []
        for wav in audio_waveforms:
            squeezed_wav = wav.squeeze()
            processed_waveforms.append(squeezed_wav)
            lengths.append(squeezed_wav.size(0))

        batch_inputs = torch.nn.utils.rnn.pad_sequence(processed_waveforms, batch_first=True, padding_value=0.0)

        #Create Masks (Vectorized)
        max_len = batch_inputs.shape[1]

        lengths_tensor = torch.tensor(lengths).unsqueeze(1) # Shape: (Batch, 1)
        range_tensor = torch.arange(max_len).unsqueeze(0)   # Shape: (1, Max_Len)
        batch_masks = (range_tensor < lengths_tensor).long()

        #Stack Targets
        batch_targets = torch.stack(batch_targets, dim=0)

        #Stack Genders
        genders = torch.stack(genders, dim=0)

        return batch_inputs, batch_masks, batch_targets, genders

class Plot:
    @staticmethod
    def waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None, size=(12, 4)):        
        num_frames = waveform.size(0)
        waveform = waveform.numpy()
        time_axis = torch.arange(0, num_frames) / sample_rate

        plt.figure(figsize=size)
        plt.plot(time_axis, waveform)
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
    def spectrogram(spectrogram, title="Spectrogram", ylabel="Frequency bin", xlabel="Frame",
                     size=(12, 4), cmap='magma'):
        spectrogram = spectrogram.squeeze().numpy()
        plt.figure(figsize=size)
        plt.imshow(spectrogram, origin='lower', aspect='auto', cmap=cmap)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    
    @staticmethod
    def mfcc(mfcc, title="MFCC", ylabel="MFCC Coefficient", xlabel="Frame", size=(12, 4)):
        mfcc = mfcc.squeeze().numpy()
        plt.figure(figsize=size)
        plt.imshow(mfcc, origin='lower', aspect='auto', interpolation='nearest', cmap='cividis')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.colorbar()
        plt.show()

class Utils:
    @staticmethod
    def load_4_torch(path):
        audio, sample_rate = audioflux.read(path=path)
        audio = torch.from_numpy(audio).float()
        return audio
    
    @staticmethod
    def load_as_np(path):
        audio, sample_rate = audioflux.read(path=path)
        return (audio, sample_rate)

    @staticmethod
    def load_resample_as_np(path, target_sample_rate):
        audio, sample_rate = audioflux.read(path=path)
        if sample_rate != target_sample_rate:
            audio = audioflux.resample(audio, sample_rate, target_sample_rate)
        return (audio, target_sample_rate)
    