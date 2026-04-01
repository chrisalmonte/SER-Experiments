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
TEST_NAME = 'TEST'
LOAD_CHECKPOINT = 100
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

# PASTE MODEL STRUCTURE AND SET MODEL MANAGER

model.to(device)
model_manager.set_model(model, "", None)
model_manager.load_checkpoint(f"{model_manager.model_directory}/checkpoints/{LOAD_CHECKPOINT}", for_inference=True)

# PASTE DATSET LOADER AND TEST LOOP. 

# SAVE RESULTS TO VARIABLE

# Save
log.log_properties(f"Test_results ({TEST_NAME})", results)
log.save()
log.save_txt()
