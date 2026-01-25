import torch
from torch import nn

class MaskedAvgPool1d(nn.Module):
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
        lengths = lengths.unsqueeze(1).float().to(x.device)
        lengths = torch.clamp(lengths, min=1.0)
        
        return sum_features / lengths
    