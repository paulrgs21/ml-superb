import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import HubertModel
from peft import LoraConfig, get_peft_model


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


class HuBERT_CTC_LoRA(nn.Module):
    """
    Complete ML-SUPERB architecture:
    1. HuBERT-base (frozen)
    2. Weighted sum of layers
    3. SpecAugment
    4. Conv downsample ÷2
    5. Transformer (2 layers, 8 heads, 1024 FFN)
    6. CTC head
    """
    
    def __init__(self, vocab_size, model_name='utter-project/mHuBERT-147-base-3rd-iter', lora_r = 8, lora_alpha = 16, lora_dropout = 0.1, target_modules = None):
        super().__init__()
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # 1. Load encoder
        base_encoder = HubertModel.from_pretrained(model_name)

        # Freeze all base encoder weights
        for param in base_encoder.parameters():
            param.requires_grad = False

        # 2. Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        self.encoder = get_peft_model(base_encoder, lora_config)

        # 3. Encoder config
        num_layers = self.encoder.config.num_hidden_layers + 1
        hidden_size = self.encoder.config.hidden_size

        # 4. Same downstream as baseline
        self.weighted_sum = WeightedSumLayer(num_layers)

        self.spec_augment = SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            num_freq_masks=2,
            num_time_masks=2,
        )

        self.downsample = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2,
        )

        self.ctc_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, audios, audio_lengths=None):
        outputs = self.encoder(
            audios,
            output_hidden_states=True,
        )

        all_layers = outputs.hidden_states
        hidden = self.weighted_sum(all_layers)
        hidden = self.spec_augment(hidden)

        hidden = hidden.transpose(1, 2)
        hidden = self.downsample(hidden)
        hidden = hidden.transpose(1, 2)

        hidden = self.transformer(hidden)
        hidden = self.dropout(hidden)
        logits = self.ctc_head(hidden)

        return logits, hidden.shape[1]  # (batch, time, vocab), output lengths

