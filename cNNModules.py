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
    

#CNN-based Neural Network for regression tasks
class CNNRaw2VAD(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(),
            )

        self.frontend = nn.Sequential(
            # 16kHz ~10ms = 160
            nn.Conv1d(1, 32, kernel_size=400, stride=10, padding=200),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            conv_block(32, 64, 15, 2),
            conv_block(64, 128, 15, 2),
            conv_block(128, 256, 15, 2),
        )
        
        self.masked_pool = MaskedAvgPool1D()

        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, input, audio_lengths):
        features = self.frontend(input)
        features = self.encoder(features)

        frontend_steps = (audio_lengths // 10) + 1
        downsampled_lengths = (frontend_steps + 7) // 8
        downsampled_lengths = torch.clamp(downsampled_lengths, min=1, max=features.size(-1))

        pooled_embedding = self.masked_pool(features, downsampled_lengths)        
        logits = self.regression_head(pooled_embedding)
        return logits
    