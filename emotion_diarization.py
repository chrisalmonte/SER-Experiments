import torch
import torch.nn as nn
import audioflux
#import getopt
#import sys
import librosa
import numpy as np
from enum import StrEnum
from transformers import WavLMModel
from peft import PeftModel, get_peft_model, set_peft_model_state_dict

# Custom modules
import cNNModules
import cLogger

# region ------ CONFIG ---------
AUDIO_PATH = r"C:\Datasets\ravdess\Audio_Speech_Actors_01-24\Actor_01\03-01-07-02-01-01-01.wav"

ANALYSIS_CONFIG = {
    #Seconds
    "window_type": "hann",
    "window_size": 0.5,
    "stride_size": 0.25,
    "batch_size": 6,
}

CLASS_MODEL = {
    "adapter_path": "output/models/Animacronica/class/lora_adapter",
    "pooling module": cNNModules.LayerWeightedAvgPooling(13),
    "head_path": "output/models/Animacronica/class/training_state.pt",
    "output_map": {
        0: 'Neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        6: 'Surprise'
    },
}

VAD_MODEL = {
    "adapter_path": "output/models/Animacronica/vad/lora_adapter",
    "pooling module": cNNModules.LayerAutoPooling(),
    "head_path": "output/models/Animacronica/vad/training_state.pt",
    "output_map": {
        0: 'Valence',
        1: 'Activation',
        2: 'Dominance'
    },
}

INT_MODEL = {
    "adapter_path": "output/models/Animacronica/intensity/lora_adapter",
    "pooling module": cNNModules.LayerAutoPooling(),
    "head_path": "output/models/Animacronica/intensity/training_state.pt",
    "output_map": {
        0: 'Normal',
        1: 'Strong',
    },
}
# endregion

class Mode(StrEnum):
    CLASS = "Emotion Class"
    VAD = "VAD"
    INT = "Intensity"

# region Commnand-line argument parsing
#args = sys.argv[1:]
#options = "i:o:w:s:"
#long_options = ["Input=", "Output=", "Window=", "Stride="]
#
#try:
#    arguments, values = getopt.getopt(args, options, long_options)
#    for currentArg, currentVal in arguments:
#        if currentArg in ("-i", "--Input"):
#            print("Input file:", currentVal)
#        elif currentArg in ("-o", "--Output"):
#            print("Output mode:", currentVal)
#        elif currentArg in ("-w", "--Window"):
#            print("Window size:", currentVal)
#        elif currentArg in ("-s", "--Stride"):
#            print("Stride:", currentVal)
#except getopt.error as err:
#    print(str(err))
#
#if len(sys.argv) > 1:
#    audio_path = sys.argv[1]
#    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else window_size
#    stride = int(sys.argv[3]) if len(sys.argv) > 3 else stride
# endregion

# region -----MODEL DEFINITION-----
class EmotionAnalysisMT(nn.Module):
    def __init__(self, class_components, vad_components, int_components, base_model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        
        base_model = WavLMModel.from_pretrained(base_model_name)
        self.wavl_emo = PeftModel.from_pretrained(base_model, class_components["adapter_path"], adapter_name=Mode.CLASS)
        self.wavl_emo.load_adapter(vad_components["adapter_path"], adapter_name=Mode.VAD)
        self.wavl_emo.load_adapter(int_components["adapter_path"], adapter_name=Mode.INT)

        self.class_pooling =class_components["pooling module"]
        self.class_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(256, 7),
        )
        state = torch.load(class_components["head_path"], weights_only=True)
        self.class_pooling.load_state_dict(state['custom_heads']['encoder_pooling'])
        self.class_head.load_state_dict(state['custom_heads']['regression_head'])

        self.vad_pooling = vad_components["pooling module"]
        self.vad_head = nn.Sequential(
            nn.Linear(1536, 812),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),

            nn.Linear(812, 360),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),

            nn.Linear(360, 120),
            nn.LeakyReLU(0.01),
            nn.Linear(120, 3),
        )
        state = torch.load(vad_components["head_path"], weights_only=True)
        self.vad_pooling.load_state_dict(state['custom_heads']['encoder_pooling'])
        self.vad_head.load_state_dict(state['custom_heads']['regression_head'])

        self.int_pooling = int_components["pooling module"]
        self.int_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),

            nn.Linear(256, 2),
        )
        state = torch.load(int_components["head_path"], weights_only=True)
        self.int_pooling.load_state_dict(state['custom_heads']['encoder_pooling'])
        self.int_head.load_state_dict(state['custom_heads']['regression_head'])

    def frame_statistical_pooling(self, features, attention_masks=None):
        if attention_masks is not None:
            downsample_factor = 320
            mask_downsampled = attention_masks[:, ::downsample_factor]
            m = mask_downsampled.unsqueeze(1).unsqueeze(-1).float()
            feat_len = features.size(2)
            if m.size(2) != feat_len:
                if m.size(2) > feat_len:
                    m = m[:, :, :feat_len, :]
                else:
                    pad = feat_len - m.size(2)
                    m = torch.nn.functional.pad(m, (0, 0, 0, pad))

            masked_features = features * m
            valid_frame_sum = m.sum(dim=2).clamp(min=1e-9)
            mean_pooled = masked_features.sum(dim=2) / valid_frame_sum
            sq_diff = (features - mean_pooled.unsqueeze(2)) ** 2
            sq_diff = sq_diff * m
            var_pooled = sq_diff.sum(dim=2) / valid_frame_sum
            std_pooled = torch.sqrt(var_pooled + 1e-9)
        else:
            mean_pooled = features.mean(dim=2)
            var_pooled = ((features - mean_pooled.unsqueeze(2)) ** 2).mean(dim=2)
            std_pooled = torch.sqrt(var_pooled + 1e-9)            
        return torch.cat([mean_pooled, std_pooled], dim=-1)

    def forward(self, task, x, attention_mask=None):
        self.wavl_emo.set_adapter(task)

        outputs = self.wavl_emo(x, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        hidden_states_pooled = self.frame_statistical_pooling(hidden_states, attention_mask)
        
        match task:
            case Mode.CLASS:
                x = self.class_pooling(hidden_states_pooled)
                logits = self.class_head(x)
                return logits
            case Mode.VAD:
                x = self.vad_pooling(hidden_states_pooled, layers_dim=1)
                logits = self.vad_head(x)
                return logits
            case Mode.INT:
                x = self.int_pooling(hidden_states_pooled, layers_dim=1)
                logits = self.int_head(x)
                return logits
            case _:
                raise ValueError(f"Task '{task}' is not recognized")
# endregion

# region -----MODEL INITIALIZATION-----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

model = EmotionAnalysisMT(CLASS_MODEL, VAD_MODEL, INT_MODEL)
model.to(device)
# endregion

# region -----LOAD AUDIO-----
def batch_waveforms(audio_waveforms):
        processed_waveforms = []
        lengths = []
        for wav in audio_waveforms:
            wav = torch.from_numpy(wav).float()
            squeezed_wav = wav.squeeze()
            processed_waveforms.append(squeezed_wav)
            lengths.append(squeezed_wav.size(0))
        batch_inputs = torch.nn.utils.rnn.pad_sequence(processed_waveforms, batch_first=True, padding_value=0.0)

        #Create Masks (Vectorized)
        max_len = batch_inputs.shape[1]
        lengths_tensor = torch.tensor(lengths).unsqueeze(1) # Shape: (Batch, 1)
        range_tensor = torch.arange(max_len).unsqueeze(0)   # Shape: (1, Max_Len)
        batch_masks = (range_tensor < lengths_tensor).long()

        return batch_inputs, batch_masks

waveform, sample_rate = audioflux.read(AUDIO_PATH)
print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

if sample_rate != 16000:
    print("Resampling to 16kHz...")
    waveform = audioflux.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

#Split audio
window_size = int(ANALYSIS_CONFIG["window_size"] * sample_rate)
stride_size = int(ANALYSIS_CONFIG["stride_size"] * sample_rate)

if waveform.shape[0] > window_size:
    window_func = librosa.filters.get_window(ANALYSIS_CONFIG["window_type"], window_size).astype(np.float32)
    frames = librosa.util.frame(waveform, frame_length=window_size, hop_length=stride_size)
    windows = frames * window_func[:, np.newaxis]
    windows = list(windows.T)
else:
    windows = [waveform]
input, mask = batch_waveforms(windows)
# endregion

# region -----INFERENCE & DIARIZATION-----
predictions = [{"chunk_index": i, "keyframe_time": ((i * stride_size) + (window_size // 2)) / sample_rate} for i in range(input.shape[0])]
model.eval()

with torch.no_grad():
    iterations = len(input) // ANALYSIS_CONFIG["batch_size"] + (1 if len(input) % ANALYSIS_CONFIG["batch_size"] != 0 else 0)
    for i in range(iterations):
        start = i*ANALYSIS_CONFIG["batch_size"]
        end = (i+1)*ANALYSIS_CONFIG["batch_size"]
        batch_input = input[start:end]
        batch_mask = mask[start:end]

        for task in Mode:
            raw_predictions = model(task, batch_input.to(device), attention_mask=batch_mask.to(device))
            raw_predictions = raw_predictions.cpu().numpy()

            for chunk in range(len(raw_predictions)):
                match task:
                    case Mode.CLASS:
                        chunk_prediction = CLASS_MODEL["output_map"][raw_predictions[chunk].argmax()]
                        predictions[start + chunk][task.value] = chunk_prediction
                    case Mode.VAD:
                        for vad_i, value in enumerate(raw_predictions[chunk]):
                            predictions[start + chunk][VAD_MODEL['output_map'][vad_i]] = value
                    case Mode.INT:
                        chunk_prediction = INT_MODEL["output_map"][raw_predictions[chunk].argmax()]
                        predictions[start + chunk][task.value] = chunk_prediction

for pred in predictions:
    print(pred)

#Diarization logic (placeholder)
