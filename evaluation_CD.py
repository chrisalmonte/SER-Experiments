#Imports
import pickle
import pandas as pd
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

TEST_NAME = 'over_MSP_Test1'
model_manager = ModelManager('output\models\wavlemo\class_6', new_run=False)
#LOAD_CHECKPOINT = 'best'
APPEND_RESULTS_TO_LOG = False

if APPEND_RESULTS_TO_LOG:
    with open(APPEND_RESULTS_TO_LOG, 'rb') as file:
        log = pickle.load(file)
else:
    log = cLogger.Log(model_manager.model_directory, prefix=f"TEST_{TEST_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.log_message(f"Test {TEST_NAME} Device: {device.type}")

results = None
model = None

# PASTE MODEL STRUCTURE AND SET MODEL MANAGER


from transformers import WavLMModel, WavLMConfig
from peft import LoraConfig, get_peft_model

class_params = {
    "output_map": {
        0: 'Neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        #6: 'Surprise',
        #7: 'Calm',
    },
    #Label map only used to remap strings in dataframes. May be None
    "label_map": {
        'N': 0, # Neutral
        'H': 1, # Happiness
        'S': 2, # Sadness
        'A': 3, # Anger
        'F': 4, # Fear
        'D': 5, # Disgust
        #'U': 6, # Surprise
        #'Ca':0  # Calm as neutral
    },
}

dataframe_params = {
    "labels_train_path": "/home/imd-temp/datasets/crema-d/labels/cremad_labels_folds.csv",
    "map_labels": ("EmoClass", class_params["label_map"]),
    "train_partition": [('Fold', [2,3,4])],
    "dev_partition": [('Fold', [1])],
    "test_partition": [('Fold', [5])],
}

loader_params = {
    "dataset_target_column": "EmoClass",
    "dataset_file_column": "FileName",
    "dataset_subdir_column": "Directory",
    "batch_size": 8,
    "batch_size_test": 4,
    "shuffle_train": True,
    "collate_function": cAudiotools.Collate.waveform_dynamic_wMasks,
    "data_transform": None,
    "target_transform": None,
    "pin_memory": True,
    "num_workers": 2,
    "persistent_workers": True,
}

dataset_params = {
    "main_dir": "/home/imd-temp/datasets",
    "target_column": "EmoClass",
    "filename_column": "FileName",
    "subdir_column": "Directory",
    "resample": False,
    "target_sample_rate": 16000,
}



df_test = pd.read_csv("output/processing/custom_labels/mspp2/divided_labels_consensus_fs.csv")
df_test = df_test[df_test["Split_Set"].isin(["Test1"])]
main_dir = r"C:\Datasets\_compiled"

dataset_test = cAudiotools.ClassDFSubdirAudioDataset(
    df_test,
    main_dir,
    dataset_params["target_column"],
    subdir_column_name=dataset_params["subdir_column"],
    name_column_name=dataset_params["filename_column"],
    transform=None,
    target_transform=loader_params["target_transform"],
    resample=dataset_params["resample"],
    target_sample_rate=dataset_params["target_sample_rate"],
    include_only=('EmoClass', class_params["label_map"].keys()),
    map_dict=class_params["label_map"]
    )
dataset_test_loader = DataLoader(
    dataset_test,
    batch_size=loader_params["batch_size_test"],
    shuffle=False,
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    )

#print number of samples in test set
log.log_message(f"Test set samples: {len(dataset_test)}")
if len(dataset_test) == 0:
    #stop the program
    SystemExit("No samples in test set. Check dataframe filters and paths.")

wavlm_params = {
    "model_name": "microsoft/wavlm-base-plus",
    "use_spec_augment": True,
}

config = WavLMConfig.from_pretrained(
    wavlm_params["model_name"],
    use_spec_augment=wavlm_params["use_spec_augment"])

wavlm_backbone = WavLMModel.from_pretrained(wavlm_params["model_name"], config=config)

class NeuralNetwork(nn.Module):
    def __init__(self, base_model):
        super().__init__()        

        self.wavlm = base_model
        self.hidden_size = self.wavlm.config.hidden_size

        lora_config = LoraConfig(
            r=8,                     # Rank
            lora_alpha=16,           # Scaling factor
            target_modules=["q_proj", "v_proj"], # Inject LoRA into the Attention layers
            lora_dropout=0.1,        # Dropout specifically for the LoRA weights
            bias="none",
        )

        self.wavlm = get_peft_model(self.wavlm, lora_config)

        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size*2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(256, len(class_params["output_map"]))
        )

        #self.encoder_pooling = cNNModules.LayerWeightedAvgPooling(self.wavlm.config.num_hidden_layers + 1)
        self.encoder_pooling = cNNModules.LayerAutoPooling()
    
    def frame_statistical_pooling(self, features, attention_masks=None):
        #Features shape: (Batch, Layers, Frames, Hidden_Size)

        if attention_masks is not None:
            #-----Mask downsampling------

            # WavLM base+ stride product
            downsample_factor = 320
            #Downsample masks to match WavLM outputs
            mask_downsampled = attention_masks[:, ::downsample_factor]
            # Reshape to broadcast across Layers and Hidden_Size
            # (Batch, 1, Mask Downsampled frames) -> (Batch, 1, Mask Downsampled frames, 1)
            m = mask_downsampled.unsqueeze(1).unsqueeze(-1).float()

            # Ensure length matches features (in case of rounding differences)
            # Features shape: (batch, layers, frames, hidden)
            feat_len = features.size(2)
            if m.size(2) != feat_len:
                # Pad or truncate (rare, but safe)
                if m.size(2) > feat_len:
                    m = m[:, :, :feat_len, :]
                else:
                    pad = feat_len - m.size(2)
                    m = torch.nn.functional.pad(m, (0, 0, 0, pad))

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
            # Population mean
            mean_pooled = features.mean(dim=2)
            # Population variance: E[(X - mean)^2]
            var_pooled = ((features - mean_pooled.unsqueeze(2)) ** 2).mean(dim=2)
            std_pooled = torch.sqrt(var_pooled + 1e-9)

        # Concatenate mean and std
        # -> (Batch, Pooled Layers, Hidden_Size * 2)            
        return torch.cat([mean_pooled, std_pooled], dim=-1)

    def forward(self, input, attention_masks):
        ssl_output = self.wavlm(input, attention_mask=attention_masks, output_hidden_states=True)        
        hidden_states = torch.stack(ssl_output.hidden_states, dim=1)
        # Shape: (Batch, Layers, Time, Hidden)        
        utterance_raw = self.frame_statistical_pooling(hidden_states, attention_masks)
        # Shape: (Batch, Layers, Hidden * 2)
        #utterance_weighted = self.encoder_pooling(utterance_raw)
        utterance_weighted = self.encoder_pooling(utterance_raw, layers_dim=1)
        # Shape: (Batch, Hidden)
        logits = self.regression_head(utterance_weighted)
        return logits

model = NeuralNetwork(wavlm_backbone).to(device)



###

model_manager.set_model(model, "", None)
model_manager.load_checkpoint(model_manager.model_directory, for_inference=True)

# PASTE DATASET LOADER AND TEST LOOP.



from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassConfusionMatrix, MulticlassRecall
import pandas as pd

#Test loop
def test_loop(dataloader, model, num_classes, output_map, device, pinned_memory=False):
    model.eval()

    accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    with torch.no_grad():
        for inputs, masks, targets in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory)

            with torch.amp.autocast('cuda'):
                logits = model(inputs, masks)
                pred = torch.argmax(logits, dim=1)

            accuracy.update(pred, targets)
            f1_macro.update(pred, targets)
            precision.update(pred, targets)
            recall.update(pred, targets)
            conf_matrix.update(pred, targets)

        matrix = conf_matrix.compute().cpu().numpy()
        class_names = [output_map[i] for i in range(num_classes)]
        df = pd.DataFrame(matrix, index=class_names, columns=class_names )
        df.index.name = "True \\ Pred"
        matrix_str = df.to_string()

        results = {
            "Accuracy": accuracy.compute().item(),
            "F1_Score_Macro": f1_macro.compute().item(),
            "Precision_Macro": precision.compute().item(),
            "Recall_Macro": recall.compute().item(),
            "Confusion_Matrix": matrix,
            "Confusion_Matrix_Str": f"\n\n{matrix_str}\n"
        }            
        return results

log.log_message("\n********* Testing *********\n")

results = test_loop(dataset_test_loader, model, len(class_params["output_map"]), class_params["output_map"], device, pinned_memory=loader_params["pin_memory"])




# SAVE RESULTS TO results VARIABLE

# Save
log.log_properties(f"Test_results ({TEST_NAME})", results)
log.save()
log.save_txt()
