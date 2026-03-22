import pandas as pd

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import soundfile as sf

import soundfile as sf

from pathlib import Path

from functools import partial

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
            audio_dir: containing the .wav 
        """
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.reset_index(drop=True)
        else:
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
        audio_path = self.audio_dir / row["audio_filename"]

        try:
            audio, sr = sf.read(audio_path)
            waveform = torch.tensor(audio, dtype=torch.float32)

            # soundfile: mono -> (time,), multi -> (time, channels)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, time)
            else:
                waveform = waveform.transpose(0, 1)  # (channels, time)

            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return {
                "audio": waveform.squeeze(0),
                "text": row["text"],
            }

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return {
                "audio": torch.zeros(16000),
                "text": "",
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

def make_collate_fn(vocab):
    return partial(collate_fn, vocab=vocab)


def load_lang_data(lang, data_root, data_size="10min"):
    """
    Load train/dev/test DataFrames for one language.
    Adds columns: audio_path (full path to .wav), lang.
    """
    lang_dir = Path(data_root) / lang
    audio_dir = lang_dir / "wav"

    suffix = "10min" if data_size == "10min" else "1h"
    df_train = parse_transcript_file(lang_dir / f"transcript_{suffix}_train.txt")
    df_dev = parse_transcript_file(lang_dir / "transcript_10min_dev.txt")
    df_test = parse_transcript_file(lang_dir / "transcript_10min_test.txt")

    for df in [df_train, df_dev, df_test]:
        fix_audio_extension(df)
        df["audio_path"] = df["audio_filename"].apply(lambda f: str(audio_dir / f))
        df["lang"] = lang

    return df_train, df_dev, df_test


class MultilingualASRDataset(Dataset):
    """
    Like ASRDataset, but each row carries its own full audio path.
    Expects df columns: [audio_path, text]
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        print(f"MultilingualASRDataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = Path(row["audio_path"])

        try:
            audio, sr = sf.read(audio_path)
            waveform = torch.tensor(audio, dtype=torch.float32)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return {"audio": waveform.squeeze(0), "text": row["text"]}

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return {"audio": torch.zeros(16000), "text": ""}