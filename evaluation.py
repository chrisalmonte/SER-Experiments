#Imports
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#Custom modules
import cAudiotools
import cLogger
import cTransforms
from cModelManagerLRA import ModelManager
import cNNModules
from enum import Enum

LOG_PATH = 'output/models/WavLM_BP_VAD_LoRa_Gender/run_2026_03_26-032822/WavLM_BP_VAD_LoRa_Gender_2026_03_27-020740.pkl'
TEST_NAME = 'Test set 1&2 (Epoch 100)'
model_manager = ModelManager('output/models/WavLM_BP_VAD_LoRa_Gender/run_2026_03_26-032822', new_run=False)

if LOG_PATH:
    with open(LOG_PATH, 'rb') as file:
        log = pickle.load(file)
else:
    log = cLogger.Log('output/logs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device.type)

results = None
model = None

class Loss(Enum):
    avg_loss_val = "Validation avg. loss"
    avg_loss_train = "Training avg. loss"

#Parameters:

shift_params = {
    "min": -0.3,
    "max": 0.3,
    "unit": "seconds",
    "prob": 0.5,
}

loader_params = {
    "dataset_dir": "/home/imd-temp/datasets",
    "dataset_train_labels": '/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_train_u_3000.csv',
    "dataset_labels": "/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_consensus.csv",
    "dataset_train_partition": ("Split_Set", ["Train"]),
    "dataset_dev_partition": ("Split_Set", ["Development"]),
    "dataset_test_partition": ("Split_Set", ["Test1", "Test2"]),
    "batch_size": 2,
    "batch_size_test": 8,
    "shuffle_train": True,
    "collate_function": cAudiotools.Collate.waveform_dynamic_wMasks_gender,
    "data_transform": cTransforms.ShiftSample(**shift_params),
    "target_transform": None,
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": True,
}

training_params = {
    "epochs": 50,
    "loss_function": nn.MSELoss(),
    "checkpoint_interval": 5,
    "checkpoint_before_training": False,
    "criterion_for_best": Loss.avg_loss_val.value,
}

grad_acumulation_params = {
    "use_grad_accumulation": True,
    "simulated_batch_size": 32,
}

wavlm_params = {
    "model_name": "microsoft/wavlm-base-plus",
    "use_spec_augment": True,
    "mask_time_prob": 0.05,    # 5% of the time steps will be masked
    "mask_time_length": 10,    # Each mask will be 10 frames long (approx 0.2 seconds)
    "mask_feature_prob": 0.05, # 5% of the frequency/feature dimensions will be masked
    "mask_feature_length": 10  # Each mask covers 10 feature channels
}

optimizer_params = {    
    "LoRA_learning_rate": 1e-4,
    "LoRA_adam_betas": (0.9, 0.98),
    "LoRA_adam_epsilon": 1e-8,
    "LoRA_weight_decay": 1e-4,
    "Regressor_learning_rate": 1e-3,
    "Regressor_adam_betas": (0.9, 0.98),
    "Regressor_adam_epsilon": 1e-8,
    "Regressor_weight_decay": 1e-4,
    "Pooling_learning_rate": 1e-3,
    "Pooling_adam_betas": (0.9, 0.98),
    "Pooling_adam_epsilon": 1e-8,
    "Pooling_weight_decay": 0,
}

scheduler_params = {
    "use_scheduler": False,
    "eta_min": 1e-5,
}

# -------------------------- Create data loaders --------------------------


#Test set
dataset_test = cAudiotools.VADSubdirAudioDatasetGender(
    loader_params["dataset_labels"],
    loader_params["dataset_dir"],
    ("EmoVal", "EmoAct", "EmoDom"),
    "Gender",
    transform=None,
    target_transform=loader_params["target_transform"],
    subdir_column_name="Directory",
    name_column_name="FileName",
    include_only=loader_params["dataset_test_partition"],
    gender_map_dict={"Male": 0, "Female": 1, "Unknown": 2}
    )
dataset_test_loader = DataLoader(
    dataset_test,
    batch_size=loader_params["batch_size_test"],
    shuffle=False,
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    num_workers=loader_params["num_workers"],
    persistent_workers=loader_params["persistent_workers"],
    )

# --------------------------- Define model -------------------------------
from transformers import WavLMModel, WavLMConfig
from peft import LoraConfig, get_peft_model

config = WavLMConfig.from_pretrained(
    wavlm_params["model_name"],
    use_spec_augment=wavlm_params["use_spec_augment"],
    mask_time_prob=wavlm_params["mask_time_prob"],
    mask_time_length=wavlm_params["mask_time_length"],
    mask_feature_prob=wavlm_params["mask_feature_prob"],
    mask_feature_length=wavlm_params["mask_feature_length"],
    )

wavlm_backbone = WavLMModel.from_pretrained(wavlm_params["model_name"], config=config)

class NeuralNetwork(nn.Module):
    def __init__(self, base_model):
        super().__init__()        

        self.wavlm = base_model
        self.hidden_size = self.wavlm.config.hidden_size

        lora_config = LoraConfig(
            r=16,                     # Rank
            lora_alpha=16,           # Scaling factor
            target_modules=["q_proj", "v_proj"], # Inject LoRA into the Attention layers
            lora_dropout=0.1,        # Dropout specifically for the LoRA weights
            bias="none",
        )

        self.wavlm = get_peft_model(self.wavlm, lora_config)

        self.regression_head = nn.Sequential(
            nn.Linear((self.hidden_size*2) + 3, 812),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(812, 360),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(360, 120),
            nn.LeakyReLU(),
            
            nn.Linear(120, 3)
        )

        self.encoder_pooling = cNNModules.LayerAutoPooling()

        self.wavlm_norm = nn.LayerNorm(self.hidden_size * 2)
    
    def frame_statistical_pooling(self, features, attention_masks=None):
        #Features shape: (Batch, Layers, Frames, Hidden_Size)

        if attention_masks is not None:
            #-----Mask downsampling------

            # Number of downsampled frames
            feat_len = features.size(2)            
            # Add dimensions for interpolation: 
            # (Batch, Mask Original frames) -> (Batch, 1, Mask Original frames)
            m = attention_masks.unsqueeze(1).float()
            # Downsample mask using Nearest Neighbor
            # (Batch, Mask Original frames) -> (Batch, 1, Mask Downsampled frames)
            m = nn.functional.interpolate(m, size=feat_len, mode='nearest')
            # Reshape to broadcast across Layers and Hidden_Size
            # (Batch, 1, Mask Downsampled frames) -> (Batch, 1, Mask Downsampled frames, 1)
            m = m.unsqueeze(-1).float()

            #-------Apply Mask------- 
            
            # (Batch,   1,    Mask Downsampled frames,  1           ) v
            # (Batch, Layers,       Frames,            Hidden_Size  )
            masked_features = features * m

            #-----Pooling----------
            valid_frame_sum = m.sum(dim=2).clamp(min=1e-9)

            # Mean
            # Masked features: (Batch, Layers, Masked Frames, Hidden_Size))
            mean_pooled = masked_features.sum(dim=2) / valid_frame_sum
            # Mean pooled: (Batch, Mean Pooled Layers, Hidden_Size)

            # Variance/Standard Deviation
            # (Batch, Layers            , Frames, Hidden_Size) - v 
            # (Batch, Mean Pooled Layers,  1    , Hidden_Size) ** 2
            sq_diff = (features - mean_pooled.unsqueeze(2)) ** 2
            sq_diff = sq_diff * m
            # Sq_diff: (Batch, Deviation Layers , Hidden_Size)
            var_pooled = sq_diff.sum(dim=2) / valid_frame_sum
            std_pooled = torch.sqrt(var_pooled + 1e-9)
            # (Batch, Pooled Layers, Hidden_Size) for mean and std
        else:
            mean_pooled = features.mean(dim=2)
            std_pooled = features.std(dim=2)
            # (Batch, Pooled Layers, Hidden_Size) for mean and std

        # Concatenate mean and std
        # -> (Batch, Pooled Layers, Hidden_Size * 2)            
        return torch.cat([mean_pooled, std_pooled], dim=-1)

    def forward(self, input, attention_masks, gender):
        ssl_output = self.wavlm(input, attention_mask=attention_masks, output_hidden_states=True)        
        hidden_states = torch.stack(ssl_output.hidden_states, dim=1)
        # Shape: (Batch, Layers, Time, Hidden)        
        utterance_raw = self.frame_statistical_pooling(hidden_states, attention_masks)
        # Shape: (Batch, Layers, Hidden * 2)
        utterance_weighted = self.encoder_pooling(utterance_raw, layers_dim=1)
        utterance_norm = self.wavlm_norm(utterance_weighted)
        # Shape: (Batch, Hidden)
        # Concatenate gender information with the weighted utterance
        utterance_gender= torch.cat([utterance_norm, gender], dim=1)
        logits = self.regression_head(utterance_gender)
        return logits

model = NeuralNetwork(wavlm_backbone)

from torchmetrics.regression import ConcordanceCorrCoef, MeanSquaredError, MeanAbsoluteError

#Test loop
def test_loop(dataloader, model, device, pinned_memory=False):
    model.eval()

    concordance = ConcordanceCorrCoef(num_outputs=3).to(device)
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    with torch.no_grad():
        for inputs, masks, targets, genders in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory)
            genders = genders.to(device, non_blocking=pinned_memory)

            with torch.amp.autocast('cuda'):
                pred = model(inputs, masks, genders)

            concordance.update(pred, targets)
            mse.update(pred, targets)
            mae.update(pred, targets)

        results = {
            "Concordance_Correlation_Coefficient": concordance.compute().cpu().tolist(),
            "Mean_Squared_Error": mse.compute().item(),
            "Mean_Absolute_Error": mae.compute().item()
        }
            
        return results

model.to(device)
model_manager.set_model(model, "", None)
model_manager.load_checkpoint(f"{model_manager.model_directory}/checkpoints/epoch_100", for_inference=True)

results = test_loop(dataset_test_loader, model, device, pinned_memory=loader_params["pin_memory"])

# Save
log.log_properties(f"Test_results ({TEST_NAME})", results)
log.save()
log.save_txt()
