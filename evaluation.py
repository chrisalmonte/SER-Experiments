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

TEST_NAME = 'ClassRVS_over_MSP2'
model_manager = ModelManager('output\models\WavLM_BP_Class_LoRa_RVS\Desp_I\F5', new_run=False)
LOAD_CHECKPOINT = 'best'
APPEND_RESULTS_TO_LOG = 'output\models\WavLM_BP_Class_LoRa_RVS\Desp_I\F5\Extra_Tests.pkl'

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

model_manager.set_model(model, "", None)
model_manager.load_checkpoint(f"{model_manager.model_directory}/checkpoints/{LOAD_CHECKPOINT}", for_inference=True)

# PASTE DATASET LOADER AND TEST LOOP.

# SAVE RESULTS TO results VARIABLE

# Save
log.log_properties(f"Test_results ({TEST_NAME})", results)
log.save()
log.save_txt()
