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
    