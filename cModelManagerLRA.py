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
        self.scheduler = None
        self.loss_property_key = None
        
        # Track best metrics
        self.best_model_metrics = {}
        self.best_model_epoch = -1
    
    def set_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion_key: str, scheduler = None, best='min'):
        self.model = model
        self.optimizer = optimizer
        self.loss_property_key = criterion_key
        self.scheduler = scheduler
        self.minimize = best == 'min'

    def checkpoint(self, epoch: int, metrics, custom_name: str = None):
        dir_name = custom_name if custom_name else f"epoch_{epoch}"
        checkpoint_dir = os.path.join(self.model_directory, "checkpoints", dir_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA Adapters (Uses PEFT's native save)
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        self.model.wavlm.save_pretrained(lora_dir)
        
        # Save Custom Heads, Optimizer, and Training Metadata
        heads_state = {
            'encoder_pooling': self.model.encoder_pooling.state_dict(),
            'regression_head': self.model.regression_head.state_dict()
        }

        training_state = {
            'epoch': epoch,
            'custom_heads': heads_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics
        }
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, state_path)
        
        print(f"Checkpoint saved at {checkpoint_dir}")
    
    def check_best(self, epoch, metrics):
        """Checks if current epoch is the best, and if so, saves immediately to disk."""
        if self.loss_property_key not in metrics:
            raise KeyError(f"The specified loss property key '{self.loss_property_key}' is not in the metrics.")
        
        best = self.best_model_metrics.get(self.loss_property_key, math.inf if self.minimize else -math.inf)
        current = metrics[self.loss_property_key]
        
        if (current < best and self.minimize) or (current > best and not self.minimize):            
            self.best_model_epoch = epoch
            self.best_model_metrics = deepcopy(metrics)
            print(f"New best model at epoch: {epoch}")
            self.checkpoint(self.best_model_epoch, self.best_model_metrics, custom_name="best")

    def load_checkpoint(self, checkpoint_dir: str, for_inference: bool = False):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_dir}")
        if not self.model:
            raise ValueError("Model must be set before loading a checkpoint.")
        if not for_inference and not self.optimizer:
            raise ValueError("Optimizer must be set before loading.")
               
        # Load Custom Heads & Optimizer
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        state = torch.load(state_path, weights_only=for_inference)

        if not for_inference:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            if self.scheduler and state.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        epoch = state['epoch']
        metrics = state['metrics']
        self.model.encoder_pooling.load_state_dict(state['custom_heads']['encoder_pooling'])
        self.model.regression_head.load_state_dict(state['custom_heads']['regression_head'])
        
        # Load LoRA Adapters
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")):
            from safetensors.torch import load_file
            lora_state = load_file(os.path.join(lora_dir, "adapter_model.safetensors"))
        else:
            lora_state = torch.load(os.path.join(lora_dir, "adapter_model.bin"), weights_only=True)
        set_peft_model_state_dict(self.model.wavlm, lora_state)
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
        if for_inference:
            self.model.eval()
            print('Model set to evaluation mode.')
        else:
            self.model.train()
            print('Model set to training mode.')
        return epoch, metrics
    
    def load_best_metrics(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No training state found at {checkpoint_path}")
        
        state = torch.load(checkpoint_path, weights_only=True)        
        self.best_model_epoch = state['epoch']
        self.best_model_metrics = state['metrics']
        print(f"Best model metrics loaded from {checkpoint_path}: {self.best_model_metrics}")
        return self.best_model_epoch, self.best_model_metrics

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