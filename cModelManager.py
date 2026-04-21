from copy import deepcopy
import math
import os
import torch
from datetime import datetime

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
        self.best_model_metrics = {}
        self.best_model_state_dict = None
        self.best_model_optim_state_dict = None
        self.best_model_epoch = -1
    
    def set_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion_key: str):
        self.model = model
        self.optimizer = optimizer
        self.loss_property_key = criterion_key

    def checkpoint(self, epoch: int, metrics):
        path = os.path.join(self.model_directory, "checkpoints", f"checkpoint_epoch_{epoch}.tar")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")
    
    def save_for_inference(self, name="final"):
        path = os.path.join(self.model_directory, f"{name}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved for inference at {path}")
    
    def check_best(self, epoch, metrics):
        if self.loss_property_key not in metrics:
            raise KeyError(f"The specified loss property key '{self.loss_property_key}' is not in the metrics.")
        
        best = self.best_model_metrics.get(self.loss_property_key, math.inf)
        current = metrics[self.loss_property_key]
        if current < best:            
            self.best_model_epoch = deepcopy(epoch)
            self.best_model_metrics = deepcopy(metrics)
            self.best_model_state_dict = deepcopy(self.model.state_dict())
            self.best_model_optim_state_dict = deepcopy(self.optimizer.state_dict())
            print(f"New best model at epoch: {epoch}")
    
    def save_best(self, save_for_inference=False):
        checkpoint_path = os.path.join(self.model_directory, "checkpoints", "best.tar")
        checkpoint = {
            'epoch': self.best_model_epoch,
            'model_state_dict': self.best_model_state_dict,
            'optimizer_state_dict': self.best_model_optim_state_dict,
            'metrics': self.best_model_metrics
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        if save_for_inference:
            inf_path = os.path.join(self.model_directory, "best.pt")
            torch.save(self.best_model_state_dict, inf_path)
            print(f"Best model saved for inference at {inf_path}")

    def load_checkpoint(self, path):
        if not self.model or not self.optimizer:
            raise ValueError("Model and optimizer must be set before loading a checkpoint.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
               
        checkpoint = torch.load(path, weights_only=False)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        metrics = checkpoint['metrics']
        self.model.train()
        print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
        return epoch, metrics
    
    def load(self, path: str):
        if not self.model or not self.optimizer:
            raise ValueError("Model and optimizer must be set before loading a checkpoint.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
                
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        print(f"Model loaded for inference from {path}")

    def load_best(self, from_checkpoint: bool = False):        
        if from_checkpoint:
            checkpoint_path = os.path.join(self.model_directory, "checkpoints", "best.tar")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            metrics = checkpoint['metrics']
            self.model.train()
            print(f"Best model checkpoint loaded from {checkpoint_path}, which was from epoch {epoch}")
        else:
            inf_path = os.path.join(self.model_directory, "best.pt")
            if not os.path.isfile(inf_path):
                raise FileNotFoundError(f"No model found at {inf_path}")

            self.model.load_state_dict(torch.load(inf_path, weights_only=True))
            self.model.eval()
            print(f"Best model loaded for inference from {inf_path}")

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
            
        