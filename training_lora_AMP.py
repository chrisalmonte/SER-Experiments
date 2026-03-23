##----------------------Huggingface login---------------------
import sys
if len(sys.argv) > 1:
    from huggingface_hub import login
    print("Attempting to log to Huggingface Hub.\n")
    login(token=sys.argv[1])

##------------------------Model Properties-----------------------
#Imports
from enum import Enum
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#Custom modules
import cAudiotools
import cLogger
import cTransforms
import cModelManagerLRA
import cNNModules

MODEL_NAME = "WavLM_L_VAD_LoRa"
MODELS_DIR = "/home/imd-temp/projects/SER-Experiments/output/models"
model_description = """
WavLM large finetuned using LoRA for VAD reggression on MSP-podcast 2.
Key Differences:
 + Statistical pooling as frame pooling.
 + Time shifting, masking and frequency masking. 
 + VAD output values range is 1 to 7.
 + 2 Hidden Layers to regression head, with LeakyReLU.
"""

#Define output paths
model_mngr = cModelManagerLRA.ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
log = cLogger.Log(model_mngr.model_directory, prefix=MODEL_NAME)
log.log_property("model_name", MODEL_NAME)
log.log_property("model_description", model_description, show=False)
log.log_property("model_dir", model_mngr.model_directory, show=False)

#Torch properties and device
log.log_property("torch_version", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    log.log_property("device", "cuda")
    log.log_property("GPU_count", torch.cuda.device_count())
    log.log_property("GPU_device", torch.cuda.get_device_name(0))
else:
    log.log_property("device", "cpu")


#-------------------------- Define parameters --------------------------
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
log.log_properties("Shift Augmentation", shift_params, show=False)

loader_params = {
    "dataset_dir": "/home/imd-temp/datasets",
    "dataset_train_labels": '/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_train_u_3000.csv',
    "dataset_labels": "/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_consensus.csv",
    "dataset_train_partition": ("Split_Set", ["Train"]),
    "dataset_dev_partition": ("Split_Set", ["Development"]),
    "dataset_test_partition": ("Split_Set", ["Test1"]),
    "batch_size": 2,
    "shuffle_train": True,
    "collate_function": cAudiotools.Collate.waveform_dynamic_wMasks,
    "data_transform": cTransforms.ShiftSample(**shift_params),
    "target_transform": None,
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": True,
}
log.log_properties("Loader", loader_params, show=False)

training_params = {
    "epochs": 50,
    "loss_function": nn.MSELoss(),
    "checkpoint_interval": 5,
    "checkpoint_before_training": False,
    "criterion_for_best": Loss.avg_loss_val.value,
}
log.log_properties("Training", training_params, show=False)

grad_acumulation_params = {
    "use_grad_accumulation": True,
    "simulated_batch_size": 32,
}
log.log_properties("Gradient Accumulation", grad_acumulation_params, show=False)

wavlm_params = {
    "model_name": "microsoft/wavlm-large",
    "use_spec_augment": True,
    "mask_time_prob": 0.05,    # 5% of the time steps will be masked
    "mask_time_length": 10,    # Each mask will be 10 frames long (approx 0.2 seconds)
    "mask_feature_prob": 0.05, # 5% of the frequency/feature dimensions will be masked
    "mask_feature_length": 10  # Each mask covers 10 feature channels
}
log.log_properties("WavLM", wavlm_params, show=False)

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
log.log_properties("Optimizer Parameters", optimizer_params, show=False)

scheduler_params = {
    "use_scheduler": False,
    "eta_min": 1e-5,
}
log.log_properties("Scheduler Parameters", scheduler_params, show=False)


# -------------------------- Create data loaders --------------------------
#Train set
dataset_train = cAudiotools.VADSubdirAudioDataset(
    loader_params["dataset_train_labels"],
    loader_params["dataset_dir"],
    ("EmoVal", "EmoAct", "EmoDom"),
    transform=loader_params["data_transform"],
    target_transform=loader_params["target_transform"],
    subdir_column_name="Directory",
    name_column_name="FileName",
    include_only=loader_params["dataset_train_partition"]
    )
dataset_train_loader = DataLoader(
    dataset_train,
    batch_size=loader_params["batch_size"],
    shuffle=loader_params["shuffle_train"],
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    num_workers=loader_params["num_workers"],
    persistent_workers=loader_params["persistent_workers"],
    )

#Development (validation) set
dataset_dev = cAudiotools.VADSubdirAudioDataset(
    loader_params["dataset_labels"],
    loader_params["dataset_dir"],
    ("EmoVal", "EmoAct", "EmoDom"),
    transform=None,
    target_transform=loader_params["target_transform"],
    subdir_column_name="Directory",
    name_column_name="FileName",
    include_only=loader_params["dataset_dev_partition"]
    )
dataset_dev_loader = DataLoader(
    dataset_dev,
    batch_size=loader_params["batch_size"],
    shuffle=False,
    collate_fn=loader_params["collate_function"],
    pin_memory=loader_params["pin_memory"],
    num_workers=loader_params["num_workers"],
    persistent_workers=loader_params["persistent_workers"],
    )

#Test set
dataset_test = cAudiotools.VADSubdirAudioDataset(
    loader_params["dataset_labels"],
    loader_params["dataset_dir"],
    ("EmoVal", "EmoAct", "EmoDom"),
    transform=None,
    target_transform=loader_params["target_transform"],
    subdir_column_name="Directory",
    name_column_name="FileName",
    include_only=loader_params["dataset_test_partition"]
    )
dataset_test_loader = DataLoader(
    dataset_test,
    batch_size=loader_params["batch_size"],
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
            nn.Linear(self.hidden_size*2, 612),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(612, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 3)
        )

        self.encoder_pooling = cNNModules.LayerAutoPooling()
    
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

    def forward(self, input, attention_masks):
        ssl_output = self.wavlm(input, attention_mask=attention_masks, output_hidden_states=True)        
        hidden_states = torch.stack(ssl_output.hidden_states, dim=1)
        # Shape: (Batch, Layers, Time, Hidden)        
        utterance_raw = self.frame_statistical_pooling(hidden_states, attention_masks)
        # Shape: (Batch, Layers, Hidden * 2)
        utterance_weighted = self.encoder_pooling(utterance_raw, layers_dim=1)
        # Shape: (Batch, Hidden)
        logits = self.regression_head(utterance_weighted)
        return logits

model = NeuralNetwork(wavlm_backbone).to(device)
log.log_property("model_structure", str(model))

loss_fn = training_params["loss_function"]

optimizer = torch.optim.AdamW([
    # LoRA for WavLM
    {'params': [p for p in model.wavlm.parameters() if p.requires_grad], 
     'lr': optimizer_params["LoRA_learning_rate"],
     'betas': optimizer_params["LoRA_adam_betas"],
     'eps': optimizer_params["LoRA_adam_epsilon"], 
     'weight_decay': optimizer_params["LoRA_weight_decay"],},
    
    # Layer Pooling  
    {'params': model.encoder_pooling.parameters(), 
     'lr': optimizer_params["Pooling_learning_rate"],
     'betas': optimizer_params["Pooling_adam_betas"],
     'eps': optimizer_params["Pooling_adam_epsilon"], 
     'weight_decay': optimizer_params["Pooling_weight_decay"],},
    
    # Regression Head 
    {'params': model.regression_head.parameters(), 
     'lr': optimizer_params["Regressor_learning_rate"],
     'betas': optimizer_params["Regressor_adam_betas"],
     'eps': optimizer_params["Regressor_adam_epsilon"], 
     'weight_decay': optimizer_params["Regressor_weight_decay"],}     
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=training_params["epochs"], 
    eta_min=scheduler_params["eta_min"]
)


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


#---------------------------------- Training ------------------------------------
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

    test_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_val.value] = test_loss


#Set Model Manager
model_mngr.set_model(model, optimizer, training_params["criterion_for_best"])

#Training
epoch_metrics = {Loss.avg_loss_train.value: math.inf, Loss.avg_loss_val.value: math.inf}
remaining_for_checkpoint = training_params["checkpoint_interval"]

if training_params["checkpoint_before_training"]:
        model_mngr.checkpoint(0, epoch_metrics)
        log.save()

log.track_time(True, message="Starting training.")
total_epochs = training_params["epochs"]
pinned_memory = loader_params["pin_memory"]

log.log_message("\n********* Training *********\n")

for epoch in range(total_epochs):
    log.log_message(f"Epoch {epoch + 1} of {total_epochs}...")
    
    if grad_acumulation_params["use_grad_accumulation"]:
        train_loop_grad_accumulation(
            dataset_train_loader, model, loss_fn, optimizer, loader_params["batch_size"], grad_acumulation_params["simulated_batch_size"], 
            metrics_dict=epoch_metrics, pinned_memory=pinned_memory)
    else:
        train_loop(dataset_train_loader, model, loss_fn, optimizer, metrics_dict=epoch_metrics, pinned_memory=pinned_memory)
    
    log.log_message("Validation")
    validation_loop(dataset_dev_loader, model, loss_fn, metrics_dict=epoch_metrics, pinned_memory=pinned_memory)

    log.log_epoch(epoch + 1, epoch_metrics)
    log.save()

    if scheduler_params["use_scheduler"]:
        scheduler.step()
    
    #Checkpointing
    remaining_for_checkpoint -= 1
    if remaining_for_checkpoint == 0:
        model_mngr.checkpoint(epoch + 1, epoch_metrics)
        remaining_for_checkpoint = training_params["checkpoint_interval"]
    
    #Save best model
    model_mngr.check_best(epoch + 1, epoch_metrics)

#Save last checkpoint if not saved at the end of training
if training_params["epochs"] % training_params["checkpoint_interval"] != 0:
    model_mngr.checkpoint(total_epochs, epoch_metrics)

log.log_elapsed_time(message="\n Training completed \n")
log.track_time(False, show=False)
log.log_properties("Last_epoch", epoch_metrics)
log.log_properties("Best_model", model_mngr.best_model_metrics | {"epoch": model_mngr.best_model_epoch})
log.save()
log.plot_epoch_values(save_path=f'{model_mngr.model_directory}/epoch_values.png')


#----------------------------- Evaluation -------------------------------
from torchmetrics.regression import ConcordanceCorrCoef, MeanSquaredError, MeanAbsoluteError

#Test loop
def test_loop(dataloader, model, device, pinned_memory=False):
    model.eval()

    concordance = ConcordanceCorrCoef(num_outputs=3).to(device)
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    with torch.no_grad():
        for inputs, masks, targets in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory)

            with torch.amp.autocast('cuda'):
                pred = model(inputs, masks)

            concordance.update(pred, targets)
            mse.update(pred, targets)
            mae.update(pred, targets)

        results = {
            "Concordance_Correlation_Coefficient": concordance.compute().cpu().tolist(),
            "Mean_Squared_Error": mse.compute().item(),
            "Mean_Absolute_Error": mae.compute().item()
        }
            
        return results

log.log_message("\n********* Testing *********\n")

for mode in ["Final", "Best"]:
    if mode == "Best":
        model_mngr.load_checkpoint(f"{model_mngr.model_directory}/checkpoints/best", for_inference=True)    
    log.log_message(f"Evaluating model ({mode})...")
    results = test_loop(dataset_test_loader, model, device, pinned_memory=loader_params["pin_memory"])
    log.log_properties(f"Test_results ({mode})", results)
    
log.save()
log.save_txt()
