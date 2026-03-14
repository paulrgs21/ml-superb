import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device, grad_accum=1):
    """Train one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        audios = batch['audios'].to(device)
        texts = batch['texts'].to(device)
        audio_lengths = batch['audio_lengths']
        text_lengths = batch['text_lengths']
        
        # Forward
        logits,T = model(audios, audio_lengths)
        
        # CTC loss expects log_probs in (T, N, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, N, C)
        
        # Input lengths (time steps after encoder)
        input_lengths = torch.full(
            (log_probs.size(1),),
            T,
            dtype=torch.long
        )
        
        # Compute loss
        loss = criterion(log_probs, texts, input_lengths, text_lengths)
        loss = loss / grad_accum  # Scale for accumulation
        
        # Backward
        loss.backward()
        
        # Update every grad_accum steps
        if (batch_idx + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f"{loss.item() * grad_accum:.4f}"})

    if len(loader) % grad_accum != 0 :
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate on dev/test set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for batch in pbar:
            audios = batch['audios'].to(device)
            texts = batch['texts'].to(device)
            audio_lengths = batch['audio_lengths']
            text_lengths = batch['text_lengths']
            
            logits,T = model(audios, audio_lengths)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            input_lengths = torch.full(
                (log_probs.size(1),),
                T,
                dtype=torch.long
            )
            
            loss = criterion(log_probs, texts, input_lengths, text_lengths)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, path, dev_loss=None, config=None, vocab=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dev_loss': dev_loss,
        'config': config,
        'vocab': vocab,
    }, path)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.1f}%)")
    print(f"Frozen params: {(total_params - trainable_params) / 1e6:.1f}M")
    
    return total_params, trainable_params

