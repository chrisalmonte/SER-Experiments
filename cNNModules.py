import torch
from torch import nn

class MaskedAvgPool1D(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, lengths):
        #Create mask
        max_len = x.size(2)
        mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(1)
        
        # Mask input
        x_masked = x * mask
        sum_features = x_masked.sum(dim=2)

        valid_counts = mask.sum(dim=2).clamp_min(1).float()
        
        return sum_features / valid_counts

class LayerAutoPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, layers_dim=0):
        # x shape: (Batch, Layers, Hidden)        
        alpha_x = self.alpha * x
        weights = nn.functional.softmax(alpha_x, dim=layers_dim)
        # (Batch, Hidden)        
        output = (x * weights).sum(dim=layers_dim)
        
        return output

class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets):
        # Calculate across the batch (dim=0) for each output feature (V, A, D)
        pred_mean = torch.mean(predictions, dim=0)
        target_mean = torch.mean(targets, dim=0)
        
        pred_var = torch.var(predictions, dim=0, unbiased=False)
        target_var = torch.var(targets, dim=0, unbiased=False)
        
        # Vectorized Covariance
        covariance = torch.mean((predictions - pred_mean) * (targets - target_mean), dim=0)
        
        # CCC Calculation
        numerator = 2 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean)**2
        
        ccc = numerator / (denominator + self.eps)
        
        return torch.mean(1.0 - ccc)
    
    