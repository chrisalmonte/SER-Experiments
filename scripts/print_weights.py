import torch

WEIGHTS_FILE = 'output/models/WavLM_BP_Class_LoRa_Re_RVS/IF_S_Bias/Fold_6/checkpoints/best/training_state.pt'
TARGET_MODULE = 'custom_heads'
TARGET_SUBMODULE = 'encoder_pooling'
SAVE_TO_TXT = None

state_dict = torch.load(WEIGHTS_FILE, map_location=torch.device('cpu'), weights_only=True)

if TARGET_MODULE in state_dict:
    module = state_dict[TARGET_MODULE]
    weights = module[TARGET_SUBMODULE] if TARGET_SUBMODULE else module
    print(f"--- Weights for {TARGET_MODULE} ---")
    print(weights)
    print(f"Shape: {weights.shape}")
        
    if SAVE_TO_TXT:
        with open(SAVE_TO_TXT, 'w') as f:
            f.write(str(weights.numpy()))
else:
    print(f"Key '{TARGET_MODULE}' not found.")
    print("Here are the available keys in your state_dict:")
    for key in state_dict.keys():
        print(key)