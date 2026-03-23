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

LOG_PATH = 'output/models/WavLM_L_VAD_LoRa/run_2026_03_21-234007/log.pkl'
TEST_NAME = 'TEST'
model_manager = ModelManager('/home/imd-temp/projects/SER-Experiments/output/models/WavLM_L_VAD_LoRa/run_2026_03_21-234007', new_run=False)

if LOG_PATH:
    with open(LOG_PATH, 'rb') as file:
        log = pickle.load(file)
else:
    log = cLogger.Log('output/logs')

results = None

# PASTE MODEL STRUCTURE AND SET MODEL MANAGER

# PASTE DATSET LOADER AND TEST LOOP. 

# SAVE RESULTS TO LOG

# Save
log.log_properties(f"Test_results ({TEST_NAME})", results)
log.save(overwrite=False)
log.save_txt()
