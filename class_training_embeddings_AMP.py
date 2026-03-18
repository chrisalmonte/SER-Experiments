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

MODEL_NAME = "emotion2vec_Class"
MODELS_DIR = "/home/imd-temp/projects/SER-Experiments/output/models"
model_description = "Emotion class Regression using emotion2vec as a feature extractor."

#Define output paths
model_mngr = cModelManager.ModelManager(f"{MODELS_DIR}/{MODEL_NAME}")
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
loader_params = {
    "dataset_labels": "/home/imd-temp/datasets/msp-podcast-2_divided/labels/divided_labels_consensus.csv",
    "embeddings_dir": "/home/imd-temp/datasets/msp-podcast-2_divided/e2v_embeddings",
    "dataset_train_partition": ("SpkrID", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    "dataset_dev_partition": ("SpkrID", [21,22,23,24]),
    "dataset_test_partition": ("SpkrID", [16,17,18,19,20]),
    "batch_size": 64,
    "shuffle_train": True,
    "collate_function": None,
    "data_transform": None,
    "target_transform": None,
    "pin_memory": True,
    "num_workers": 4,
    "persistent_workers": True,
}
log.log_properties("Loader", loader_params, show=False)

label_map = {
    'N': 0, # Other
    'H': 1, # Happiness
    'S': 2, # Sadness
    'A': 3, # Anger
    'F': 4, # Fear
    'D': 5, # Disgust
    'U': 6  # Surprise
}
log.log_property("label_map", label_map, show=False)

output_map = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Sadness',
    3: 'Anger',
    4: 'Fear',
    5: 'Disgust',
    6: 'Surprise'
}
log.log_property("output_map", output_map, show=False)

training_params = {
    "epochs": 50,
    "checkpoint_interval": 6,
    "checkpoint_before_training": False,
    "criterion_for_best": Loss.avg_loss_val.value,
}
log.log_properties("Training", training_params, show=False)

grad_acumulation_params = {
    "use_grad_accumulation": False,
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

scheduler_params = {
    "use_scheduler": True,
    "eta_min": 1e-5,
}
log.log_properties("Scheduler", scheduler_params, show=False)


# -------------------------- Create data loaders --------------------------
#Train set
dataset_train = cAudiotools.ClassEmbeddingsDataset(
    loader_params["dataset_labels"],
    loader_params["embeddings_dir"],
    "EmoClass",
    mappings_dict=label_map,
    transform=loader_params["data_transform"],
    target_transform=loader_params["target_transform"],
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
dataset_dev = cAudiotools.ClassEmbeddingsDataset(
    loader_params["dataset_labels"],
    loader_params["embeddings_dir"],
    "EmoClass",
    mappings_dict=label_map,
    transform=loader_params["data_transform"],
    target_transform=loader_params["target_transform"],
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
dataset_test = cAudiotools.ClassEmbeddingsDataset(
    loader_params["dataset_labels"],
    loader_params["embeddings_dir"],
    "EmoClass",
    mappings_dict=label_map,
    transform=loader_params["data_transform"],
    target_transform=loader_params["target_transform"],
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
inputs, targets = sample_batch

log.log_message(f"Inputs Shape: {inputs.shape}")
log.log_message(f"Targets Shape: {targets.shape}")
log.log_message(f"Output range: Min={targets.min():.2f}, Max={targets.max():.2f}")


# --------------------------- Define model -------------------------------
from cNNModules import CCCLoss

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.regression_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 7)
        )

    def forward(self, input):
        logits = self.regression_head(input)
        return logits


model = NeuralNetwork().to(device)
log.log_property("model_structure", str(model))

loss_fn = nn.CrossEntropyLoss()
log.log_property("loss_function", str(loss_fn))

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=optimizer_params["learning_rate"], 
    betas=optimizer_params["adam_betas"],
    eps=optimizer_params["adam_epsilon"],
    weight_decay=optimizer_params["weight_decay"],
    )
log.log_property("optimizer", str(optimizer))


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=training_params["epochs"], 
    eta_min=scheduler_params["eta_min"]
)
if scheduler_params["use_scheduler"]:
    log.log_property("scheduler", str(scheduler))


#---------------------------------- Training ------------------------------------
#Loop definitions
def train_loop(dataloader, model, loss_fn, optimizer, metrics_dict=None, pinned_memory=False):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    
    for batch, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory).long()

        with torch.amp.autocast('cuda'):
            pred = model(inputs)
            loss = loss_fn(pred, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = simulated_batch_size // batch_size
    
    for batch, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device, non_blocking=pinned_memory)
        targets = targets.to(device, non_blocking=pinned_memory).long()

        with torch.amp.autocast('cuda'):
            pred = model(inputs)
            loss = loss_fn(pred, targets)
            #Normalize gradients
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (batch + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        #Print progress every 100 batches
        #if batch % 100 == 0:
        #    current = batch * len(inputs)
        #    real_loss = loss.item() * accumulation_steps
        #    print(f"[{current:>6d}/{size:>6d}] batch loss: {real_loss:>7f}")

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
        for inputs, targets  in dataloader:
            inputs = inputs.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory).long()

            with torch.amp.autocast('cuda'):
                pred = model(inputs)
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

model_mngr.save_for_inference()
model_mngr.save_best(save_for_inference=True)


#----------------------------- Evaluation -------------------------------
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
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
        for inputs, targets in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, non_blocking=pinned_memory)
            targets = targets.to(device, non_blocking=pinned_memory).long()

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            accuracy.update(preds, targets)
            f1_macro.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            conf_matrix.update(preds, targets)

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
            "Confusion_Matrix": matrix_str
        }            
        return results

log.log_message("\n********* Testing *********\n")

num_classes = len(label_map.keys())
for mode in ["Final", "Best"]:
    if mode == "Best":
        model_mngr.load_best()    
    log.log_message(f"Evaluating model ({mode})...")
    results = test_loop(dataset_test_loader, model, num_classes, output_map, device, pinned_memory=loader_params["pin_memory"])
    log.log_properties(f"Test_results ({mode})", results)
    
log.save()
log.save_txt()
