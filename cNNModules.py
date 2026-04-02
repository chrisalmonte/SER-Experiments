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
    
class LayerWeightedAvgPooling(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        # Initialize with 1's at the start
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x):
        # x shape: (Batch, Layers, Hidden)
        
        norm_weights = nn.functional.softmax(self.weights, dim=0)
        
        # Reshape to (1, weights, 1) for broadcasting
        norm_weights = norm_weights.unsqueeze(0).unsqueeze(-1)
        
        # Multiply by weights and sum across the layers dimension
        output = (x * norm_weights).sum(dim=1)
        
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
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        alpha: 1D Tensor of weights for each class (optional).
        gamma: Focusing parameter to penalize hard-to-classify examples.
        reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Register alpha as a buffer so it automatically moves to the correct device 
        # (CPU/GPU) along with the model.
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs shape: (Batch_size, Num_classes)
        # targets shape: (Batch_size)

        # 1. Compute the standard Cross-Entropy Loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # 2. Get the probability of the true class (pt)
        # Since Cross-Entropy is -log(pt), we can get pt by applying exp(-CE)
        pt = torch.exp(-ce_loss)

        # 3. Compute the Focal Loss modulating factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Apply the alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = focal_loss * alpha_t

        # 5. Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    