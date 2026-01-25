import torch
import torchaudio

class Functions:
    @staticmethod
    def mono(waveform):
        if waveform.size(0) > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    @staticmethod
    def resample(waveform, orig_freq, new_freq):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(waveform)    

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
