import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import HubertModel, Wav2Vec2Model
from jiwer import cer


def load_ssl_encoder(model_name):
    """
    Auto-detect and load the correct SSL encoder class.
    HuBERT models use HubertModel, wav2vec2/XLS-R use Wav2Vec2Model.
    Both have identical interfaces for our usage.
    """
    name_lower = model_name.lower()
    model_cls = Wav2Vec2Model if (
        'wav2vec2' in name_lower or 'xls-r' in name_lower or 'xlsr' in name_lower
    ) else HubertModel

    try:
        return model_cls.from_pretrained(model_name, use_safetensors=True)
    except Exception as e:
        raise RuntimeError(
            f"Impossible to load {model_name} with safetensors."
        ) from e


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
        stacked = torch.stack(hidden_states, dim=0)
        normed_weights = F.softmax(self.weights, dim=0)
        weighted = torch.sum(
            stacked * normed_weights.view(-1, 1, 1, 1),
            dim=0
        )
        return weighted


class HuBERT_CTC(nn.Module):
    """
    ML-SUPERB downstream architecture:
    1. SSL encoder (frozen) — HuBERT or Wav2Vec2, auto-detected
    2. Weighted sum of layers
    3. SpecAugment
    4. Conv downsample ÷2
    5. Transformer (configurable layers, 8 heads, 1024 FFN)
    6. CTC head
    """
    
    def __init__(self, vocab_size, model_name='utter-project/mHuBERT-147-base-3rd-iter',
                 num_transformer_layers=2):
        super().__init__()
        
        # 1. Encoder (frozen)
        self.encoder = load_ssl_encoder(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        num_layers = self.encoder.config.num_hidden_layers + 1
        hidden_size = self.encoder.config.hidden_size
        
        # 2. Weighted sum of layers
        self.weighted_sum = WeightedSumLayer(num_layers)
        
        # 3. SpecAugment
        self.spec_augment = SpecAugment(
            freq_mask_param=27,
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
        
        # 5. Transformer (configurable depth)
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
            num_layers=num_transformer_layers
        )
        
        # 6. CTC head
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, audios, audio_lengths=None):
        outputs = self.encoder(audios, output_hidden_states=True)
        all_layers = outputs.hidden_states
        
        hidden = self.weighted_sum(all_layers)
        hidden = self.spec_augment(hidden)
        
        hidden = hidden.transpose(1, 2)
        hidden = self.downsample(hidden)
        hidden = hidden.transpose(1, 2)
        
        hidden = self.transformer(hidden)
        
        hidden = self.dropout(hidden)
        logits = self.ctc_head(hidden)
        
        return logits, hidden.shape[1]


class HuBERT_Linear_CTC(nn.Module):
    """
    Minimal downstream: SSL encoder (frozen) → Weighted sum → Linear CTC.
    No SpecAugment, no downsample, no contextual layers.
    Measures raw SSL representation quality.
    """
    
    def __init__(self, vocab_size, model_name='utter-project/mHuBERT-147-base-3rd-iter'):
        super().__init__()
        
        self.encoder = load_ssl_encoder(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        num_layers = self.encoder.config.num_hidden_layers + 1
        hidden_size = self.encoder.config.hidden_size
        
        self.weighted_sum = WeightedSumLayer(num_layers)
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, audios, audio_lengths=None):
        outputs = self.encoder(audios, output_hidden_states=True)
        all_layers = outputs.hidden_states
        
        hidden = self.weighted_sum(all_layers)
        logits = self.ctc_head(hidden)
        
        return logits, hidden.shape[1]