import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import HubertModel
from jiwer import cer


# Model

class SpecAugment(nn.Module):
    """
    SpecAugment to avoid overfitting
    """
    def __init__(self, freq_mask_param=27, time_mask_param=100,
                 num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def forward(self, x):
        """
        Args: x (batch, time, freq/hidden)
        Returns: x with applied masks
        """
        if not self.training:
            return x  # No augmentation in eval
        
        batch, time, freq = x.shape
        
        x = x.clone()
        
        # Frequency masking (on hidden dimension)
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, freq - f, (1,)).item()
            x[:, :, f0:f0+f] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, time), (1,)).item()
            t0 = torch.randint(0, time - t, (1,)).item()
            x[:, t0:t0+t, :] = 0
        
        return x


class WeightedSumLayer(nn.Module):
    """Weighted sum of SSL layer outputs"""
    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states):
        """
        Args: hidden_states: tuple of (num_layers, batch, time, hidden)
        Returns: weighted sum (batch, time, hidden)
        """
        # Stack all layers
        stacked = torch.stack(hidden_states, dim=0)  # (num_layers, batch, time, hidden)
        
        # Normalize weights
        normed_weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum
        weighted = torch.sum(
            stacked * normed_weights.view(-1, 1, 1, 1),
            dim=0
        )  # (batch, time, hidden)
        
        return weighted


class HuBERT_CTC(nn.Module):
    """
    Complete ML-SUPERB architecture:
    1. HuBERT-base (frozen)
    2. Weighted sum of layers
    3. SpecAugment
    4. Conv downsample ÷2
    5. Transformer (2 layers, 8 heads, 1024 FFN)
    6. CTC head
    """
    
    def __init__(self, vocab_size, model_name='utter-project/mHuBERT-147-base-3rd-iter'):
        super().__init__()
        
        # 1. Encoder (frozen)
        self.encoder = HubertModel.from_pretrained(model_name)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()  # Set to eval mode
        
        # Get config
        num_layers = self.encoder.config.num_hidden_layers + 1  # +1 for input
        hidden_size = self.encoder.config.hidden_size  # 768 for mHuBERT base
        
        
        # 2. Weighted sum of layers
        self.weighted_sum = WeightedSumLayer(num_layers)
        
        # 3. SpecAugment
        self.spec_augment = SpecAugment(
            freq_mask_param=27,   # Standard SUPERB benchmark values
            time_mask_param=100,
            num_freq_masks=2,
            num_time_masks=2
        )
        
        # 4. Convolutional downsample (÷2)
        self.downsample = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        # 5. Transformer (2 layers)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )
        
        # 6. CTC head
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, audios, audio_lengths=None):
        # 1. Encode (get all layers)
        outputs = self.encoder(
            audios,
            output_hidden_states=True
        )
        
        all_layers = outputs.hidden_states
        
        # 2. Weighted sum of layers
        hidden = self.weighted_sum(all_layers)  # (batch, time, hidden)
        
        # 3. SpecAugment
        hidden = self.spec_augment(hidden)
        
        # 4. Conv downsample
        hidden = hidden.transpose(1, 2)  # (batch, hidden, time)
        hidden = self.downsample(hidden)  # (batch, hidden, time/2)
        hidden = hidden.transpose(1, 2)  # (batch, time/2, hidden)
        
        # 5. Transformer
        hidden = self.transformer(hidden)
        
        # 6. Dropout + CTC
        hidden = self.dropout(hidden)
        logits = self.ctc_head(hidden)
        
        return logits, hidden.shape[1]  # (batch, time, vocab), output lengths

