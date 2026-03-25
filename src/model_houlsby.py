import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType

from transformers import HubertModel, Wav2Vec2Model


def load_ssl_encoder(model_name):
    """Auto-detect and load HuBERT or Wav2Vec2."""
    name_lower = model_name.lower()
    if 'wav2vec2' in name_lower or 'xls-r' in name_lower or 'xlsr' in name_lower:
        return Wav2Vec2Model.from_pretrained(model_name)
    else:
        return HubertModel.from_pretrained(model_name)


def get_encoder_layers(encoder):
    """
    Return the list of transformer layers from the encoder.
    HuBERT: encoder.encoder.layers
    Wav2Vec2: encoder.encoder.layers
    Both have the same structure.
    """
    return encoder.encoder.layers


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


class HoulsbyAdapter(nn.Module):
    """Houlsby-style bottleneck adapter: x + W_up(dropout(act(W_down(x))))"""
    def __init__(self, hidden_size, bottleneck=32, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, hidden_size)

    def forward(self, x):
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up_proj(z)
        return x + z


def _patch_layer_with_houlsby(layer, hidden_size, bottleneck=32, adapter_dropout=0.1):
    """
    Patch one HuBERT/Wav2Vec2 EncoderLayer to insert adapters.
    Works for both HubertEncoderLayer and Wav2Vec2EncoderLayer
    (they share the same attribute names).
    """
    layer.attn_adapter = HoulsbyAdapter(
        hidden_size=hidden_size,
        bottleneck=bottleneck,
        dropout=adapter_dropout,
    )
    layer.ff_adapter = HoulsbyAdapter(
        hidden_size=hidden_size,
        bottleneck=bottleneck,
        dropout=adapter_dropout,
    )

    def houlsby_forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # Self-attention block
        attn_residual = hidden_states
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # Houlsby adapter after attention
        hidden_states = self.attn_adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        # Feed-forward block
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        # Houlsby adapter after FFN
        hidden_states = self.ff_adapter(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs

    layer.forward = MethodType(houlsby_forward, layer)


class HuBERT_CTC_Houlsby(nn.Module):
    """
    ML-SUPERB architecture + Houlsby adapters in the SSL encoder.
    Supports HuBERT and Wav2Vec2/XLS-R models.
    """
    def __init__(self, vocab_size, model_name="utter-project/mHuBERT-147-base-3rd-iter",
                 adapter_bottleneck=32, adapter_dropout=0.1):
        super().__init__()

        # 1. Load encoder (auto-detect type)
        self.encoder = load_ssl_encoder(model_name)

        # Freeze all original encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        num_layers = self.encoder.config.num_hidden_layers + 1

        # 2. Patch each transformer layer with Houlsby adapters
        for layer in get_encoder_layers(self.encoder):
            _patch_layer_with_houlsby(
                layer,
                hidden_size=hidden_size,
                bottleneck=adapter_bottleneck,
                adapter_dropout=adapter_dropout,
            )

        # Re-enable training only for adapter weights
        for name, param in self.encoder.named_parameters():
            if "attn_adapter" in name or "ff_adapter" in name:
                param.requires_grad = True

        # 3. Downstream (same as baseline)
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