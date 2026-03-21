import os
import math
import torch
from copy import deepcopy
from datetime import datetime
from peft import set_peft_model_state_dict

class ModelManager:
    def __init__(self, model_directory: str, new_run: bool = True):
        if new_run:
            self.run_name, self.model_directory = Directories.make_unique(model_directory)
        else:
            self.model_directory = model_directory
            self.run_name = os.path.basename(model_directory)
        self.model = None
        self.optimizer = None
        self.loss_property_key = None
        
        # Track best metrics
        self.best_model_metrics = {}
        self.best_model_epoch = -1
    
    def set_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion_key: str):
        self.model = model
        self.optimizer = optimizer
        self.loss_property_key = criterion_key

    def _get_custom_heads_state(self):
        """Helper to cleanly extract only the custom PyTorch heads."""
        return {
            'encoder_pooling': self.model.encoder_pooling.state_dict(),
            'regression_head': self.model.regression_head.state_dict()
        }

    def checkpoint(self, epoch: int, metrics):
        """Creates a folder for the epoch containing LoRA, Heads, and Optimizer states."""
        checkpoint_dir = os.path.join(self.model_directory, "checkpoints", f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 1. Save LoRA Adapters (Uses PEFT's native save)
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        self.model.wavlm.save_pretrained(lora_dir)
        
        # 2. Save Custom Heads, Optimizer, and Training Metadata
        training_state = {
            'epoch': epoch,
            'custom_heads': self._get_custom_heads_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, state_path)
        
        print(f"Checkpoint saved at {checkpoint_dir}")
    
    def save_for_inference(self, name="final"):
        inf_dir = os.path.join(self.model_directory, name)
        os.makedirs(inf_dir, exist_ok=True)
        
        # 1. Save LoRA Adapters
        self.model.wavlm.save_pretrained(os.path.join(inf_dir, "lora_adapter"))
        
        # 2. Save Custom Heads
        torch.save(self._get_custom_heads_state(), os.path.join(inf_dir, "custom_heads.pt"))
        print(f"Model saved for inference at {inf_dir}")
    
    def check_best(self, epoch, metrics):
        """Checks if current epoch is the best, and if so, saves immediately to disk."""
        if self.loss_property_key not in metrics:
            raise KeyError(f"The specified loss property key '{self.loss_property_key}' is not in the metrics.")
        
        best = self.best_model_metrics.get(self.loss_property_key, math.inf)
        current = metrics[self.loss_property_key]
        
        if current < best:            
            self.best_model_epoch = epoch
            self.best_model_metrics = deepcopy(metrics)
            print(f"New best model at epoch: {epoch}")
            
            # Save best model
            best_dir = os.path.join(self.model_directory, "checkpoints", "best")
            os.makedirs(best_dir, exist_ok=True)
            
            # 1. Save LoRA
            self.model.wavlm.save_pretrained(os.path.join(best_dir, "lora_adapter"))
            
            # 2. Save State
            training_state = {
                'epoch': self.best_model_epoch,
                'custom_heads': self._get_custom_heads_state(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.best_model_metrics
            }
            torch.save(training_state, os.path.join(best_dir, "training_state.pt"))
        

    def load_checkpoint(self, checkpoint_dir: str):
        """Resumes training from a specific epoch folder."""
        if not self.model or not self.optimizer:
            raise ValueError("Model and optimizer must be set before loading a checkpoint.")
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_dir}")
               
        # 1. Load Custom Heads & Optimizer
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        state = torch.load(state_path, weights_only=False)
        
        epoch = state['epoch']
        metrics = state['metrics']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.model.encoder_pooling.load_state_dict(state['custom_heads']['encoder_pooling'])
        self.model.regression_head.load_state_dict(state['custom_heads']['regression_head'])
        
        # 2. Load LoRA Adapters
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        self._load_lora_safely(lora_dir)
        
        self.model.train()
        print(f"Checkpoint loaded from {checkpoint_dir}, resuming from epoch {epoch}")
        return epoch, metrics
    
    def load(self, dir: str):
        """Loads a model strictly for inference (Evaluation Mode)."""
        if not self.model:
            raise ValueError("Model must be set before loading.")
        if not os.path.isdir(dir):
            raise FileNotFoundError(f"No inference directory found at {dir}")
                
        # 1. Load Custom Heads
        heads_path = os.path.join(dir, "custom_heads.pt")
        heads_state = torch.load(heads_path, weights_only=True)
        self.model.encoder_pooling.load_state_dict(heads_state['encoder_pooling'])
        self.model.regression_head.load_state_dict(heads_state['regression_head'])
        
        # 2. Load LoRA Adapters
        lora_dir = os.path.join(dir, "lora_adapter")
        self._load_lora_safely(lora_dir)

        self.model.eval()
        print(f"Model loaded for inference from {dir}")

    def load_best(self, from_checkpoint: bool = False):
        best_dir = os.path.join(self.model_directory, "checkpoints", "best")
        
        if from_checkpoint:
            return self.load_checkpoint(best_dir)
        else:
            self.load(best_dir)

    def _load_lora_safely(self, lora_dir: str):
        """Internal helper to safely inject LoRA weights regardless of safetensors or bin format."""
        if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")):
            from safetensors.torch import load_file
            lora_state = load_file(os.path.join(lora_dir, "adapter_model.safetensors"))
        else:
            lora_state = torch.load(os.path.join(lora_dir, "adapter_model.bin"), weights_only=True)
        
        set_peft_model_state_dict(self.model.wavlm, lora_state)


class Directories:
    @staticmethod
    def make_unique(directory: str):
        unique_name = f"run_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}"
        directory = os.path.join(directory, unique_name)
        if os.path.exists(directory):
            directory = directory + "_copy"
            print("The directory already exists. A new directory has been created:", directory)
        try:
            os.makedirs(directory, exist_ok=True)
            os.mkdir(os.path.join(directory, "checkpoints"))
            print("Directory created at:", directory)
        except Exception as e:
            raise OSError(f"The directory could not be created: {e}")
        return unique_name, directory