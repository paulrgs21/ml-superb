import torch
import torch.nn as nn
import torch.nn.functional as F
 
from transformers import HubertModel, Wav2Vec2Model
from peft import LoraConfig, get_peft_model
 
 
def load_ssl_encoder(model_name):
    """Auto-detect and load HuBERT or Wav2Vec2."""
    name_lower = model_name.lower()
    if 'wav2vec2' in name_lower or 'xls-r' in name_lower or 'xlsr' in name_lower:
        return Wav2Vec2Model.from_pretrained(model_name)
    else:
        return HubertModel.from_pretrained(model_name)
 
 
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=27, time_mask_param=100,
                 num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def forward(self, x):
        if not self.training:
            return x
        batch, time, freq = x.shape
        x = x.clone()
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            if f > 0 and freq - f > 0:
                f0 = torch.randint(0, freq - f, (1,)).item()
                x[:, :, f0:f0+f] = 0
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, time), (1,)).item()
            if t > 0 and time - t > 0:
                t0 = torch.randint(0, time - t, (1,)).item()
                x[:, t0:t0+t, :] = 0
        return x
 
 
class WeightedSumLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states):
        stacked = torch.stack(hidden_states, dim=0)
        normed_weights = F.softmax(self.weights, dim=0)
        weighted = torch.sum(stacked * normed_weights.view(-1, 1, 1, 1), dim=0)
        return weighted
 
 
class HuBERT_CTC_LoRA(nn.Module):
    """
    ML-SUPERB architecture + LoRA on the SSL encoder.
    Supports HuBERT and Wav2Vec2/XLS-R models.
    """
    
    def __init__(self, vocab_size, model_name='utter-project/mHuBERT-147-base-3rd-iter',
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        super().__init__()
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
 
        # 1. Load encoder (auto-detect type)
        base_encoder = load_ssl_encoder(model_name)
 
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
 
        # 4. Downstream (same as baseline)
        self.weighted_sum = WeightedSumLayer(num_layers)
        self.spec_augment = SpecAugment()
 
        self.downsample = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=2, stride=2, padding=0,
        )
 
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=1024,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
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