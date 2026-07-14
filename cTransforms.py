import torch
from audiomentations import Compose, Shift, AddGaussianSNR
from silero_vad import load_silero_vad, get_speech_timestamps

class Functions:
    @staticmethod
    def mono(waveform):
        if waveform.size(0) > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform    

class NormalizeMinus(object):
    """
    Normalizes tensor to values between -1 and 1.
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):        
        normalized = 2 * ((tensor - self.min_value) / (self.max_value - self.min_value)) - 1
        return normalized
    
class Normalize(object):
    """
    Normalizes tensor to values between 0 and 1.
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor):        
        normalized = (tensor - self.min_value) / (self.max_value - self.min_value)
        return normalized

class AudioPadTrimTo(object):
    def __init__(self, max_seconds, sample_rate=16000):
        self.max_length = max_seconds * sample_rate

    def __call__(self, waveform):
        if waveform.size(1) > self.max_length:
            return waveform[:, :self.max_length]
        else:
            padding = self.max_length - waveform.size(1)
            return torch.nn.functional.pad(waveform, (0, padding))

class ShiftSample(object):
    def __init__(self, min=-0.25, max=0.25, unit="seconds", prob=0.5, sample_rate=16000):
        self.sample_rate = sample_rate
        self.augment = Compose([
            Shift(min_shift=min, max_shift=max, shift_unit=unit, rollover=False, p=prob),
        ])

    def __call__(self, waveform_np):
        augmented_waveform = self.augment(samples=waveform_np, sample_rate=self.sample_rate)
        return augmented_waveform

class ShiftNoiseSample(object):
    def __init__(self, sft_min=-0.25, sft_max=0.25, sft_unit="seconds", sft_prob=0.5, sample_rate=16000,
                 snr_min=10, snr_max=30, snr_prob=0.5):
        self.sample_rate = sample_rate
        self.augment = Compose([
            Shift(min_shift=sft_min, max_shift=sft_max, shift_unit=sft_unit, rollover=False, p=sft_prob),
            AddGaussianSNR(min_snr_db=snr_min, max_snr_db=snr_max, p=snr_prob)
        ])

    def __call__(self, waveform_np):
        augmented_waveform = self.augment(samples=waveform_np, sample_rate=self.sample_rate)
        return augmented_waveform
    
class TrimShiftNoiseSample(object):
    def __init__(self, sft_min=-0.25, sft_max=0.25, sft_unit="seconds", sft_prob=0.5, sample_rate=16000,
                 snr_min=10, snr_max=30, snr_prob=0.5, trim_trail=0.1):
        self.vad_model = load_silero_vad()
        self.trim_trail_samples = int(trim_trail * sample_rate)
        self.sample_rate = sample_rate
        self.augment = Compose([
            Shift(min_shift=sft_min, max_shift=sft_max, shift_unit=sft_unit, rollover=False, p=sft_prob),
            AddGaussianSNR(min_snr_db=snr_min, max_snr_db=snr_max, p=snr_prob)
        ])

    def __call__(self, waveform_np):
        speech_timestamps = get_speech_timestamps(waveform_np, self.vad_model, return_seconds=False)
        if speech_timestamps is not None and len(speech_timestamps) > 0:
            start = max(0, speech_timestamps[0]['start'] - self.trim_trail_samples)
            end = min(len(waveform_np), speech_timestamps[-1]['end'] + self.trim_trail_samples)
            waveform_np = waveform_np[start:end]
        augmented_waveform = self.augment(samples=waveform_np, sample_rate=self.sample_rate)
        return augmented_waveform
