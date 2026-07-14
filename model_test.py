import librosa
import os
import audioflux
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from enum import StrEnum
from peft import PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import WavLMModel

# Custom modules
import cNNModules
from emotion_tools import AnalysisProps, EmoTask
#from cLogger import Log

TEST_NAME = "intensity_test_CREMAD"
#log = Log("output/models/wavlemo", TEST_NAME)

audio_dir = r"C:\Datasets\_compiled"
predictions = {'FileName': [], 'EmoAct': [], 'EmoVal': [], 'EmoDom': [], 'EmoInt': [], 'EmoClass': []}

    
dataset = pd.read_csv("output/processing/custom_labels/crema-d/cremad_labels.csv")
dataset = dataset[dataset['EmoInt'].isin(['low', 'medium', 'high'])]
#df['Split_Set'] = df['Split_Set'].str.strip()
#df['FileName'] = df['FileName'].str.strip()
#dataset = df[df['Split_Set'] == 'Test3']
dataset = dataset.reset_index(drop=True)

def preprocess_audio(file_path):
    waveform, sample_rate = audioflux.read(file_path)
    if sample_rate != 16000:
        waveform = audioflux.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    return waveform, sample_rate

CLASS_MODEL = {
    "adapter_path": "output/models/wavlemo/class/lora_adapter",
    "pooling module": cNNModules.LayerWeightedAvgPooling(13),
    "head_path": "output/models/wavlemo/class/training_state.pt",
    "output_map": {
        0: 'neutral',
        1: 'happiness',
        2: 'sadness',
        3: 'anger',
        4: 'fear',
        5: 'disgust',
        6: 'surprise'
    },
}

VAD_MODEL = {
    "adapter_path": "output/models/wavlemo/vad/lora_adapter",
    "pooling module": cNNModules.LayerAutoPooling(),
    "head_path": "output/models/wavlemo/vad/training_state.pt",
    "output_map": {
        0: AnalysisProps.VAL,
        1: AnalysisProps.ACT,
        2: AnalysisProps.DOM
    },
}

INT_MODEL = {
    "adapter_path": "output/models/wavlemo/intensity/lora_adapter",
    "pooling module": cNNModules.LayerAutoPooling(),
    "head_path": "output/models/wavlemo/intensity/training_state.pt",
    "output_map": {
        0: 'normal',
        1: 'strong',
    },
}
# endregion

# region -----MODEL DEFINITION-----
class EmotionAnalysisMT(nn.Module):
    def __init__(self, class_components, vad_components, int_components, base_model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        
        base_model = WavLMModel.from_pretrained(base_model_name)
        self.wavl_emo = PeftModel.from_pretrained(base_model, class_components["adapter_path"], adapter_name=EmoTask.EMOCLASS.value)
        self.wavl_emo.load_adapter(vad_components["adapter_path"], adapter_name=EmoTask.VAD.value)
        self.wavl_emo.load_adapter(int_components["adapter_path"], adapter_name=EmoTask.INTCLASS.value)

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
            case EmoTask.EMOCLASS.value:
                x = self.class_pooling(hidden_states_pooled)
                logits = self.class_head(x)
                return logits
            case EmoTask.VAD.value:
                x = self.vad_pooling(hidden_states_pooled, layers_dim=1)
                logits = self.vad_head(x)
                return logits
            case EmoTask.INTCLASS.value:
                x = self.int_pooling(hidden_states_pooled, layers_dim=1)
                logits = self.int_head(x)
                return logits
            case _:
                raise ValueError(f"Task '{task}' is not recognized")
# endregion

# region -----MODEL INITIALIZATION-----
model = None
device = None
    
    
# endregion

# region -----Requests-----
def analyze_waveform(raw_bytes, task):
    input = torch.tensor(waveform).float().unsqueeze(0)
    predictions = {}

    model.eval()

    with torch.no_grad():
        task = task.value
        raw_prediction = model(task, input.to(device), attention_mask=None)
        raw_prediction = raw_prediction.cpu().numpy()

        match task:
            case EmoTask.EMOCLASS.value:
                prediction = CLASS_MODEL["output_map"][raw_prediction.argmax()]
                predictions[AnalysisProps.CLASS] = prediction
            case EmoTask.VAD.value:
                for vad_i, value in enumerate(raw_prediction[0]):
                    predictions[VAD_MODEL['output_map'][vad_i]] = value.item()
            case EmoTask.INTCLASS.value:
                prediction = INT_MODEL["output_map"][raw_prediction.argmax()]
                predictions[AnalysisProps.INT] = prediction
    return predictions
# endregion

def preprocess_audio(file_path):
    waveform, sample_rate = audioflux.read(file_path)
    print(f"Waveform length: {waveform.shape[0]}, Sample rate: {sample_rate}")
    if sample_rate != 16000:
        print("Resampling to 16kHz...")
        waveform = audioflux.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
        print("Resampling complete.")
    return waveform, sample_rate


# EVALUATION
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    model = EmotionAnalysisMT(CLASS_MODEL, VAD_MODEL, INT_MODEL)
    model.to(device)

    print("Model loaded and ready.")   

    for idx, row in dataset.iterrows():
        print(f"Processing file {idx+1}/{len(dataset)}: {row['FileName']}")
        audio = os.path.join(audio_dir,row['Directory'], row['FileName'])
        waveform, sample_rate = preprocess_audio(audio)
        prediction = analyze_waveform(waveform, task=EmoTask.VAD)
        predictions['FileName'].append(row['FileName'])
        predictions['EmoClass'].append(row['EmoClass'] if 'EmoClass' in row else 'Unknown')
        predictions['EmoInt'].append(row['EmoInt'] if 'EmoInt' in row else 'Unknown')
        predictions['EmoAct'].append(prediction.get(AnalysisProps.ACT, 0))
        predictions['EmoVal'].append(prediction.get(AnalysisProps.VAL, 0))
        predictions['EmoDom'].append(prediction.get(AnalysisProps.DOM, 0))

    final_df = pd.DataFrame(predictions)
    print(final_df.head())
    final_df.to_csv(f"output/models/wavlemo/{TEST_NAME}_vad_predictions.csv", index=False)
    
    print("\nShutting down server and clearing memory...")
    model = None
    torch.cuda.empty_cache()

