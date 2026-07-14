# region ---------------------- Imports -------------------------------
import argparse
from enum import StrEnum
import math
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

#Custom modules
import cAudiotools
import cLogger
import cTransforms
import cModelManagerLRA
import cNNModules
from cUtils import Imbalance, DataFrames
# endregion

# region ---------------------- Argument Parser---------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WavLM for SER"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML file with model settings",
    )
    parser.add_argument(
        "--hf-key", type=str, default=None,
        help="Hugging Face token for authentication.",
    )
    return parser.parse_args()

args = parse_args()
config_path = args.config

if args.hf_key:
    from huggingface_hub import login
    print("Attempting to log to Huggingface Hub.\n")
    login(token=args.hf_key)
# endregion

# region ---------------------- Model Saving-----------------------
MODEL_NAME = "Shin_WavLM_BP_Class_RVDS"
MODELS_DIR = "output/models"
RESUME_FROM = None

model_description = ""

#Define output paths
model_mngr = cModelManagerLRA.ModelManager(f"{MODELS_DIR}/{MODEL_NAME}", new_run=not RESUME_FROM)
log = cLogger.Log(model_mngr.model_directory, prefix=MODEL_NAME)

#Torch properties and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not RESUME_FROM:
    log.log_property("config_file", cfg_path)
    log.log_property("torch_version", torch.__version__)
    log.log_property("model_name", MODEL_NAME)
    log.log_property("model_description", model_description, show=False)
    log.log_property("model_dir", model_mngr.model_directory, show=False)
    if device.type == "cuda":
        log.log_property("device", "cuda")
        log.log_property("GPU_count", torch.cuda.device_count())
        log.log_property("GPU_device", torch.cuda.get_device_name(0))
    else:
        log.log_property("device", "cpu")
# endregion   

# region ---------------------- Define parameters --------------------------
class Loss(StrEnum):
    avg_loss_val = "Validation avg. loss"
    avg_loss_train = "Training avg. loss"

class Metrics(StrEnum):
    unweighted_avg_recall = "unweighted_avg_recall"

class ClassWeighting(StrEnum):
    none = "none"
    inverse_frequency = "inverse_frequency"
    smooth_inverse_f = "smooth_inverse_frequency"
    effective_number = "effective_number"
    custom = "custom"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

class_params = {
    "output_map": {
        0: 'Neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Anger',
        4: 'Fear',
        5: 'Disgust',
        6: 'Surprise',
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
        'U': 6, # Surprise
        #'Ca':0  # Calm as neutral
    },
}

dataframe_params = {
    "labels_train_path": cfg["dataset"]["labels"],
    "map_labels": ("EmoClass", class_params["label_map"]),
    "train_partition": [('Fold', [2,3,4,5])],
    "dev_partition": [('Fold', [1])],
    "test_partition": [('Fold', [6])],
}

dataset_params = {
    "target_column": "EmoClass",
    "filename_column": "FileName",
    "subdir_column": "Directory",
    "resample": True,
    "target_sample_rate": 16000,
}

loader_params = {
    "batch_size_test": 8,
    "shuffle_train": True,
    "collate_function": cAudiotools.Collate.waveform_dynamic_wMasks,
    "data_transform": cTransforms.TrimShiftNoiseSample(**cfg["data_augmentation"]),
    "target_transform": None,
    "pin_memory": True,
    "num_workers": 0,
    "persistent_workers": False,
}

df_train, df_dev, df_test = DataFrames.make_train_dev_test(**dataframe_params)

match ClassWeighting(cfg["class_weights"]["type"]):
    case ClassWeighting.none:
        class_weights = None
        cfg["class_weights"]["weights"] = None
    case ClassWeighting.inverse_frequency:
        from sklearn.utils.class_weight import compute_class_weight
        import pandas as pd
        import numpy as np
        train_labels = df_train[dataset_params["target_column"]].values

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        cfg["class_weights"]["weights"] = class_weights.tolist()
    case ClassWeighting.smooth_inverse_f:
        class_counts_series = df_train[dataset_params["target_column"]].value_counts().sort_index()
        counts_array = class_counts_series.values
        class_weights = Imbalance.smoothed_inverse_weights(counts_array)
        cfg["class_weights"]["weights"] = class_weights.tolist()
    case ClassWeighting.effective_number:
        class_counts_series = df_train[dataset_params["target_column"]].value_counts().sort_index()
        counts_array = class_counts_series.values
        class_weights = Imbalance.effective_number_weights(counts_array, beta=cfg["class_weights"]["effective_number_beta"])
        cfg["class_weights"]["weights"] = class_weights.tolist()
    case ClassWeighting.custom:
         class_weights = torch.tensor(cfg["class_weights"]["weights"], dtype=torch.float32)

training_params = {
    "epochs": 50,
    "loss_function": nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["class_weights"]["label_smoothing"]),
    #"loss_function": cNNModules.FocalLoss(alpha=class_weights, gamma=class_weighting_params["gamma"]),
    "checkpoint_interval": 5,
    "checkpoint_before_training": False,
    "criterion_for_best": Metrics.unweighted_avg_recall.value,
}

grad_acumulation_params = {
    "use_grad_accumulation": True,
    "simulated_batch_size": 32,
}

if RESUME_FROM:
    target_epochs = training_params["epochs"]
    if target_epochs <= RESUME_FROM:
        raise ValueError(f"Target epochs ({target_epochs}) must be greater than the epoch to resume from ({RESUME_FROM}).")
    log.log_message(f"\n********** Resuming from epoch {RESUME_FROM} ********\n")
    log.log_property("new_target_epochs", target_epochs)
else:
    log.log_properties("Data Augmentation", cfg["data_augmentation"], show=False)
    log.log_properties("Classes ", class_params, show=False)
    log.log_properties("Dataframe Structure", dataframe_params, show=False)
    log.log_properties("Dataset", dataset_params, show=False)
    log.log_properties("Loader", loader_params, show=False)
    log.log_properties("Class weightings", cfg["class_weights"], show=False)
    log.log_properties("Training", training_params, show=False)
    log.log_properties("Gradient Accumulation", cfg["grad_accumulation"], show=False)
    log.log_properties("WavLM", wavlm_params, show=False)
    log.log_properties("LoRA Optimizer Parameters", cfg["lora_optimizer"], show=False)
    log.log_properties("Regressors Optimizer Parameters", cfg["regressor_optimizer"], show=False)
    log.log_properties("Pooling Optimizer Parameters", cfg["pooling_optimizer"], show=False)
    log.log_properties("Scheduler Parameters", cfg["scheduler"], show=False)
# endregion

# region ---------------------- Create data loaders --------------------------
#Train set
dataset_train = cAudiotools.ClassDFSubdirAudioDataset(
    df_train,
    cfg["dataset"]["directory"],
    dataset_params["target_column"],
    subdir_column_name=dataset_params["subdir_column"],
    name_column_name=dataset_params["filename_column"],
    transform=loader_params["data_transform"],
    target_transform=loader_params["target_transform"],
    resample=dataset_params["resample"],
    target_sample_rate=dataset_params["target_sample_rate"],
    )
dataset_train_loader = DataLoader(
    dataset_train,
    batch_size=cfg["loader"]["batch_size"],
    shuffle=loader_params["shuffle_train"],
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    num_workers=loader_params["num_workers"],
    persistent_workers=loader_params["persistent_workers"],
    )

#Development (validation) set
dataset_dev = cAudiotools.ClassDFSubdirAudioDataset(
    df_dev,
    dataset_params["main_dir"],
    dataset_params["target_column"],
    subdir_column_name=dataset_params["subdir_column"],
    name_column_name=dataset_params["filename_column"],
    transform=None,
    target_transform=loader_params["target_transform"],
    resample=dataset_params["resample"],
    target_sample_rate=dataset_params["target_sample_rate"],
    )
dataset_dev_loader = DataLoader(
    dataset_dev,
    batch_size=cfg["loader"]["batch_size"],
    shuffle=False,
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    num_workers=loader_params["num_workers"],
    persistent_workers=loader_params["persistent_workers"],
    )

#Test set
dataset_test = cAudiotools.ClassDFSubdirAudioDataset(
    df_test,
    dataset_params["main_dir"],
    dataset_params["target_column"],
    subdir_column_name=dataset_params["subdir_column"],
    name_column_name=dataset_params["filename_column"],
    transform=None,
    target_transform=loader_params["target_transform"],
    resample=dataset_params["resample"],
    target_sample_rate=dataset_params["target_sample_rate"],
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
# endregion

# region ---------------------- Define model -------------------------------
from transformers import WavLMModel, WavLMConfig
from peft import LoraConfig, get_peft_model

config = WavLMConfig.from_pretrained(
    cfg["backbone"]["model"],
    use_spec_augment=cfg["spec_augment"]["use"],
    mask_time_prob=cfg["spec_augment"]["mask_time_prob"],
    mask_time_length=cfg["spec_augment"]["mask_time_length"],
    mask_feature_prob=cfg["spec_augment"]["mask_feature_prob"],
    mask_feature_length=cfg["spec_augment"]["mask_feature_length"],
    )

wavlm_backbone = WavLMModel.from_pretrained(cfg["backbone"], config=config)
class NeuralNetwork(nn.Module):
    def __init__(self, base_model):
        super().__init__()        

        self.wavlm = base_model
        self.hidden_size = self.wavlm.config.hidden_size

        lora_config = LoraConfig(
            r=8,                                    # Rank
            lora_alpha=16,                          # Scaling factor
            target_modules=["q_proj", "v_proj"],    # Inject LoRA into the Attention layers
            lora_dropout=0.1,                       # Dropout specifically for the LoRA weights
            bias="none",
        )

        self.wavlm = get_peft_model(self.wavlm, lora_config)

        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size*2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.45),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.35),

            nn.Linear(256, len(class_params["output_map"]))
        )

        self.encoder_pooling = cNNModules.LayerWeightedAvgPooling(self.wavlm.config.num_hidden_layers + 1)
        #self.encoder_pooling = cNNModules.LayerAutoPooling()
    
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
        utterance_weighted = self.encoder_pooling(utterance_raw)
        #utterance_weighted = self.encoder_pooling(utterance_raw, layers_dim=1)
        # Shape: (Batch, Hidden)
        logits = self.regression_head(utterance_weighted)
        return logits

model = NeuralNetwork(wavlm_backbone).to(device)
loss_fn = training_params["loss_function"].to(device)

optimizer = torch.optim.AdamW([
    # LoRA for WavLM
    {'params': [p for p in model.wavlm.parameters() if p.requires_grad], 
     'lr': cfg["lora_optimizer"]["learning_rate"],
     'betas': cfg["lora_optimizer"]["adam_betas"],
     'eps': cfg["lora_optimizer"]["adam_epsilon"], 
     'weight_decay': cfg["lora_optimizer"]["weight_decay"],},
    
    # Layer Pooling  
    {'params': model.encoder_pooling.parameters(), 
     'lr': cfg["pooling_optimizer"]["learning_rate"],
     'betas': cfg["pooling_optimizer"]["adam_betas"],
     'eps': cfg["pooling_optimizer"]["adam_epsilon"], 
     'weight_decay': cfg["pooling_optimizer"]["weight_decay"],},
    
    # Regression Head 
    {'params': model.regression_head.parameters(), 
     'lr': cfg["regressor_optimizer"]["learning_rate"],
     'betas': cfg["regressor_optimizer"]["adam_betas"],
     'eps': cfg["regressor_optimizer"]["adam_epsilon"], 
     'weight_decay': cfg["regressor_optimizer"]["weight_decay"],}     
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=cfg["training"]["epochs"], 
    eta_min=cfg["scheduler"]["eta_min"]
)

if not RESUME_FROM:
    log.log_property("model_structure", str(model), show=False)


# --------------------------- Data check -------------------------------
log.log_message("\n********* Data Check *********\n")
log.log_message(f"Train samples: {len(dataset_train)}")
log.log_message(f"Dev samples: {len(dataset_dev)}")
log.log_message(f"Test samples: {len(dataset_test)}")
log.log_message(f"Batch size: {loader_params['batch_size']}")

log.log_message("\n********* Sample Batch *********\n")
sample_batch = next(iter(dataset_train_loader))
inputs, masks, targets = sample_batch

log.log_message(f"Inputs Shape: {inputs.shape}")
log.log_message(f"Targets Shape: {targets.shape}")
log.log_message(f"Masks Shape: {masks.shape}")
log.log_message(f"Input range: Min={inputs.min():.2f}, Max={inputs.max():.2f}")
log.log_message(f"Output range: Min={targets.min():.2f}, Max={targets.max():.2f}")
# endregion

# region ---------------------- Training ------------------------------------
from torchmetrics.classification import MulticlassRecall

#Loop definitions
def train_loop(dataloader, model, loss_fn, optimizer, metrics_dict=None, pinned_memory=False):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    
    for batch, (inputs, masks, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory)
        masks = masks.to(device, non_blocking=pinned_memory)

        with torch.amp.autocast('cuda'):
            pred = model(inputs, masks)
            loss = loss_fn(pred, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * inputs.size(0)
    
    epoch_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_train.value] = epoch_loss

#Loop definitions
def train_loop_grad_accumulation(dataloader, model, loss_fn, optimizer, batch_size, simulated_batch_size, metrics_dict=None, pinned_memory=False):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = simulated_batch_size // batch_size
    
    for batch, (inputs, masks, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory)
        masks = masks.to(device, non_blocking=pinned_memory)

        with torch.amp.autocast('cuda'):
            pred = model(inputs, masks)
            loss = loss_fn(pred, targets)
            #Normalize gradients
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (batch + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += (loss.item() * accumulation_steps) * inputs.size(0)
    
    # Handle any remaining gradients if the dataset size isn't divisible by accumulation_steps
    if (batch + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    epoch_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_train.value] = epoch_loss


def validation_loop(dataloader, model, loss_fn, metrics_dict=None, pinned_memory=False):
    val_uar = MulticlassRecall(num_classes=len(class_params["output_map"]), average='macro').to(device)
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for inputs, masks, targets  in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)

            with torch.amp.autocast('cuda'):
                pred = model(inputs, masks)
                loss = loss_fn(pred, targets)
            test_loss += loss.item() * inputs.size(0)
            val_uar.update(pred, targets)

    test_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_val.value] = test_loss
        metrics_dict[Metrics.unweighted_avg_recall.value] = val_uar.compute().item()

#Set Model Manager
model_mngr.set_model(model, optimizer, Metrics(cfg["training"]["criterion_for_best"]), best=cfg["training"]["criterion_mode"])

if RESUME_FROM:
    epoch_start, epoch_metrics = model_mngr.load_checkpoint(f"{model_mngr.model_directory}/checkpoints/epoch_{RESUME_FROM}")
    model_mngr.load_best_metrics(f"{model_mngr.model_directory}/checkpoints/best/training_state.pt")
else:
    epoch_start = 0
    epoch_metrics = {
        Loss.avg_loss_train.value: math.inf, 
        Loss.avg_loss_val.value: math.inf,
        }
    if training_params["checkpoint_before_training"]:
            model_mngr.checkpoint(0, epoch_metrics)
            log.save()

remaining_for_checkpoint = training_params["checkpoint_interval"]
log.track_time(True, message="Starting training.")
total_epochs = cfg["training"]["epochs"]
pinned_memory = loader_params["pin_memory"]

log.log_message("\n********* Training *********\n")
patience_counter = 0

for epoch in range(epoch_start, total_epochs):
    log.log_message(f"Epoch {epoch + 1} of {total_epochs}...")
    
    if grad_acumulation_params["use_grad_accumulation"]:
        train_loop_grad_accumulation(
            dataset_train_loader, model, loss_fn, optimizer, cfg["loader"]["batch_size"], cfg["grad_accumulation"]["simulated_batch_size"], 
            metrics_dict=epoch_metrics, pinned_memory=pinned_memory)
    else:
        train_loop(dataset_train_loader, model, loss_fn, optimizer, metrics_dict=epoch_metrics, pinned_memory=pinned_memory)
    
    log.log_message("Validation")
    validation_loop(dataset_dev_loader, model, loss_fn, metrics_dict=epoch_metrics, pinned_memory=pinned_memory)

    log.log_epoch(epoch + 1, epoch_metrics)
    log.save()

    if cfg["scheduler"]["use"]:
        scheduler.step()
    
    #Checkpointing
    remaining_for_checkpoint -= 1
    if remaining_for_checkpoint == 0:
        model_mngr.checkpoint(epoch + 1, epoch_metrics)
        remaining_for_checkpoint = training_params["checkpoint_interval"]
    
    #Save best model
    new_best = model_mngr.check_best(epoch + 1, epoch_metrics)

    if cfg["training"]["patience"]:
        if new_best:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                log.log_message(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

#Save last checkpoint if not saved at the end of training
if training_params["epochs"] % training_params["checkpoint_interval"] != 0:
    model_mngr.checkpoint(total_epochs, epoch_metrics)

log.log_elapsed_time(message="\n Training completed \n")
log.track_time(False, show=False)
log.log_properties(f"Last_epoch ({total_epochs})", epoch_metrics)
log.log_properties("Best_model", model_mngr.best_model_metrics | {"epoch": model_mngr.best_model_epoch})
log.save()
log.plot_epoch_values(save_path=f'{model_mngr.model_directory}/epoch_values_{total_epochs}.png')
# endregion

# region ---------------------- Evaluation -------------------------------
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassConfusionMatrix
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

for mode in ["Final", "Best"]:
    if mode == "Best":
        model_mngr.load_checkpoint(f"{model_mngr.model_directory}/checkpoints/best", for_inference=True)    
    log.log_message(f"Evaluating model ({mode})...")
    results = test_loop(dataset_test_loader, model, len(class_params["output_map"]), class_params["output_map"], device, pinned_memory=loader_params["pin_memory"])
    log.log_properties(f"Test_results ({mode} up to {epoch+1})", results)
    
log.save()
log.save_txt()
# endregion
