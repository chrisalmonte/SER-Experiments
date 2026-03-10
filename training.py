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
#from torchvision import transforms

#Custom modules
import cAudiotools
import cLogger
import cTransforms
import cModelManager

MODEL_NAME = "WavLM_L_VAD_LoRa"
MODELS_DIR = "/home/imd-temp/projects/SER-Experiments/output/models"
model_description = "WavLM finetuned using LoRA and avg. pooling for frame pooling and autopooling for Embedding pooling."

#Define output paths
log = cLogger.Log("/home/imd-temp/projects/SER-Experiments/output/logs", prefix=MODEL_NAME)
model_mngr = cModelManager.ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
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
loader_params = {
    "dataset_dir": "/home/imd-temp/datasets",
    "dataset_labels": "/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_consensus.csv",
    "dataset_train_partition": ("Split_Set", "Train"),
    "dataset_dev_partition": ("Split_Set", "Development"),
    "dataset_test_partition": ("Split_Set", "Test1"),
    "batch_size": 4,
    "shuffle_train": True,
    "collate_function": cAudiotools.Collate.waveform_dynamic_wMasks,
    "data_transform": None,
    "target_transform": cTransforms.NormalizeMinus(1, 7),
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": True,
}
log.log_properties("Loader", loader_params, show=False)

training_params = {
    "epochs": 30,
    "checkpoint_interval": 6,
    "checkpoint_before_training": False,
    "criterion_for_best": Loss.avg_loss_val.value,
}
log.log_properties("Training", training_params, show=False)

grad_acumulation_params = {
    "use_grad_accumulation": True,
    "simulated_batch_size": 16,
}
log.log_properties("Gradient Accumulation", grad_acumulation_params, show=False)

optimizer_params = {    
    "learning_rate": 1e-3,
    "adam_betas": (0.9, 0.999),
    "adam_epsilon": 1e-8,
    "weight_decay": 1e-4,
}
log.log_properties("Optimizer", optimizer_params, show=False)

#scheduler_params = {
#    "use_scheduler": True,
#    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(),
#}


# -------------------------- Create data loaders --------------------------
#Train set
dataset_train = cAudiotools.VADSubdirAudioDataset(
    loader_params["dataset_labels"],
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
    transform=loader_params["data_transform"],
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
    transform=loader_params["data_transform"],
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


# --------------------------- Define model -------------------------------
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

import cNNModules

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
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
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 3),
        )

        self.encoder_pooling = cNNModules.LayerAutoPooling()
    
    def frame_mean_pooling(self, features, attention_masks):        
        if attention_masks is not None:
            feat_len = features.size(2)
            
            # Add dimensions for interpolation: (Batch, Time, 1)
            m = attention_masks.unsqueeze(1).float()
            
            # Downsample mask using Nearest Neighbor
            m = nn.functional.interpolate(m, size=feat_len, mode='nearest')
            
            # Transpose to match features: (Batch, Time, 1)
            m = m.transpose(1, 2)
            m = m.unsqueeze(0)

            features = features * m
            pooled = features.sum(dim=2) / m.sum(dim=2).clamp(min=1e-9)
        else:
            pooled = features.mean(dim=2)            
        return pooled

    def forward(self, input, attention_masks):
        ssl_output = self.wavlm(input, attention_mask=attention_masks, output_hidden_states=True)
        hidden_states = torch.stack(ssl_output.hidden_states, dim=0)        
        utterance_embedding = self.frame_mean_pooling(hidden_states, attention_masks)
        pooled_embedding = self.encoder_pooling(utterance_embedding)
        logits = self.regression_head(pooled_embedding)
        return logits


model = NeuralNetwork().to(device)
log.log_property("model_structure", str(model))

loss_fn = nn.MSELoss()
log.log_property("loss_function", str(loss_fn))

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=optimizer_params["learning_rate"], 
    betas=optimizer_params["adam_betas"],
    eps=optimizer_params["adam_epsilon"],
    weight_decay=optimizer_params["weight_decay"],
    )

log.log_property("optimizer", str(optimizer))


#---------------------------------- Training ------------------------------------
#Loop definitions
def train_loop(dataloader, model, loss_fn, optimizer, metrics_dict=None, pinned_memory=False):
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    
    for batch, (inputs, masks, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory)
        masks = masks.to(device, non_blocking=pinned_memory)

        pred = model(inputs, masks)
        loss = loss_fn(pred, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        batch_loss = loss.item()

        #if batch % 100 == 0:
        #    current = batch * len(inputs)
        #    print(f"[{current:>6d}/{size:>6d}] batch loss: {batch_loss:>7f}")
    
    epoch_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_train.value] = epoch_loss

#Loop definitions
def train_loop_grad_accumulation(dataloader, model, loss_fn, optimizer, batch_size, simulated_batch_size, metrics_dict=None, pinned_memory=False):
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = simulated_batch_size // batch_size
    
    for batch, (inputs, masks, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory)
        masks = masks.to(device, non_blocking=pinned_memory)

        pred = model(inputs, masks)
        loss = loss_fn(pred, targets)
        #Normalize gradients
        loss = loss / accumulation_steps
        loss.backward()

        if (batch + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        #Print progress every 100 batches
        #if batch % 100 == 0:
        #    current = batch * len(inputs)
        #    real_loss = loss.item() * accumulation_steps
        #    print(f"[{current:>6d}/{size:>6d}] batch loss: {real_loss:>7f}")

        epoch_loss += (loss.item() * accumulation_steps) * inputs.size(0)
    
    # Handle any remaining gradients if the dataset size isn't divisible by accumulation_steps
    if (batch + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    epoch_loss /= size
    if metrics_dict:
        metrics_dict[Loss.avg_loss_train.value] = epoch_loss


def validation_loop(dataloader, model, loss_fn, metrics_dict=None, pinned_memory=False):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for inputs, masks, targets  in dataloader:
            inputs = inputs.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)

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
    
    validation_loop(dataset_dev_loader, model, loss_fn, metrics_dict=epoch_metrics, pinned_memory=pinned_memory)

    #log.log_elapsed_time(message=f"Epoch {epoch + 1} completed.")
    log.log_epoch(epoch + 1, epoch_metrics)
    log.save()
    
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

model_mngr.save_for_inference()
model_mngr.save_best(save_for_inference=True)


#----------------------------- Evaluation -------------------------------
from torchmetrics.regression import ConcordanceCorrCoef
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Test loop
def test_loop(dataloader, model, device, pinned_memory=False):
    total_predictions = []
    total_targets = []
    model.eval()

    with torch.no_grad():
        log.log_message("\n********* Testing *********\n")
        size = len(dataloader.dataset)
        for batch, (inputs, masks, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            masks = masks.to(device, non_blocking=pinned_memory)

            pred = model(inputs, masks)
            total_predictions.append(pred.cpu())
            total_targets.append(targets)
            if batch % 10 == 0:
                current = batch * len(inputs)
                print(f"[{current:>6d}/{size:>6d}] processed")
        final_predictions = torch.cat(total_predictions, dim=0)
        final_targets = torch.cat(total_targets, dim=0)
        return final_predictions, final_targets

log.log_message(f"Targets shape: {targets.shape}")
concordance = ConcordanceCorrCoef(num_outputs=3)

for mode in ["Final", "Best"]:
    if mode == "Best":
        model_mngr.load_best()
    
    log.log_message(f"Evaluating model ({mode})...")
    predictions, targets = test_loop(dataset_test_loader, model, device, pinned_memory=loader_params["pin_memory"])
    log.log_message(f"Predictions shape ({mode}): {predictions.shape}")

    results = {
        "Concordance_Correlation_Coefficient": concordance(predictions, targets).tolist(),
        "Mean_Squared_Error": mean_squared_error(targets.numpy(), predictions.numpy()),
        "Mean_Absolute_Error": mean_absolute_error(targets.numpy(), predictions.numpy())
    }
    log.log_properties(f"Test_results ({mode})", results)
    
log.save()
log.save_txt()
