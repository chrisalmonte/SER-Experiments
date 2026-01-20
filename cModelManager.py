import os
import torch
from datetime import datetime

class Directories:
    @staticmethod
    def make_unique(path: str):
        if os.path.exists(path):
            path = os.path.join(path, f"run_{datetime.now().strftime('%Y_%m_%d-%H%M%S')}")
            print("The directory already exists. A new directory has been created:", path)
        try:
            os.makedirs(path, exist_ok=True)
            os.mkdir(os.path.join(path, "checkpoints"))
            print("Directory created at:", path)
        except Exception as e:
            raise OSError(f"The directory could not be created: {e}")
        return path
            

class CheckPoint:
    @staticmethod
    def save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss, directory: str):
        path = os.path.join(directory, "checkpoints", f"checkpoint_epoch_{epoch}.tar")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")

    @staticmethod
    def load(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
        print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
        return epoch, loss
    
    @staticmethod
    def save_for_inference(model: torch.nn.Module, directory: str):
        path = os.path.join(directory, f"final.pt")
        torch.save(model.state_dict(), path)
        print(f"Model saved for inference at {path}")

    @staticmethod
    def load_for_inference(model: torch.nn.Module, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
                
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        print(f"Model loaded for inference from {path}")
        