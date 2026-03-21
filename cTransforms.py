import torch
from audiomentations import Compose, Shift

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
            Shift(min_fraction=min, max_fraction=max, shift_unit=unit, rollover=False, p=prob),
        ])

    def __call__(self, waveform_np):
        augmented_waveform = self.augment(samples=waveform_np, sample_rate=self.sample_rate)
        return augmented_waveform
