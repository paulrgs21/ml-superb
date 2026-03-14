import pandas as pd

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from pathlib import Path

def parse_transcript_file(transcript_path):
    """
    Input: utt_id audio_filename transcription
    Output: DataFrame with cols [utt_id, audio_filename, text]
    """
    data = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=2)
            utt_id, audio_filename, text = parts
            data.append({
                'utt_id': utt_id,
                'audio_filename': audio_filename,
                'text': text.lower().strip()
            })
            
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} samples from {transcript_path}")
    return df



# If audio extensions = .mp3, change it to .wav:
def fix_audio_extension(df):
    """
    Reconstruct the name: utt_id + ".wav"
    """
    df['audio_filename'] = df['utt_id'] + ".wav"
    return df




class ASRDataset(Dataset):
    """Dataset with audio and text"""
    
    def __init__(self, csv_path, audio_dir):
        """
        Args:
            csv_path: path to CSV (utt_id, audio_filename, text)
            audio_dir: file containing the .wav
        """
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Check that files do exist
        missing = 0
        for idx in range(min(10, len(self.df))):  # Check first 10
            audio_path = self.audio_dir / self.df.iloc[idx]['audio_filename']
            if not audio_path.exists():
                missing += 1
                if missing == 1:  # Print first missing file
                    print(f"File not found: {audio_path}")
        
        if missing > 0:
            print(f"Warning: {missing}/10 first files not found. Check audio_dir path!")
        else:
            print(f"Audio files found in {audio_dir}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # full audio path
        audio_path = self.audio_dir / row['audio_filename']
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return {
                'audio': waveform.squeeze(0),
                'text': row['text'],
            }
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if error
            return {
                'audio': torch.zeros(16000),
                'text': '',
            }
        


def build_vocab(dataset):
    """Build character-level vocabulary"""
    chars = set()
    for i in range(len(dataset)):
        chars.update(dataset[i]['text'])
    
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<blank>': 2,  # CTC blank
    }
    
    for char in sorted(chars):
        if char == ' ' or char.strip():
            vocab[char] = len(vocab)
    
    return vocab

def build_vocab_from_df(df):
    chars = set()

    for text in df["text"]:
        chars.update(text)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<blank>": 2,
    }

    for char in sorted(chars):
        if char == " " or char.strip():
            vocab[char] = len(vocab)

    return vocab

def collate_fn(batch, vocab):
    """Collate with padding for variable-length sequences"""
    
    # Extract audios and texts
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Audio lengths
    audio_lengths = torch.tensor([len(a) for a in audios])
    
    # Pad audios
    audios_padded = torch.nn.utils.rnn.pad_sequence(
        audios,
        batch_first=True,
        padding_value=0
    )
    
    # Encode texts
    texts_encoded = []
    for text in texts:
        encoded = [vocab.get(c, vocab['<unk>']) for c in text]
        texts_encoded.append(torch.tensor(encoded))
    
    # Text lengths
    text_lengths = torch.tensor([len(t) for t in texts_encoded])
    
    # Pad texts
    texts_padded = torch.nn.utils.rnn.pad_sequence(
        texts_encoded,
        batch_first=True,
        padding_value=vocab['<pad>']
    )
    
    return {
        'audios': audios_padded,
        'audio_lengths': audio_lengths,
        'texts': texts_padded,
        'text_lengths': text_lengths,
    }