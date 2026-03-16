import torch
from tqdm.auto import tqdm
import json
import pandas as pd


def ctc_decode(logits, idx_to_char, blank_id):
    """Simple greedy CTC decoder."""
    pred_ids = torch.argmax(logits, dim=-1)

    decoded = []
    for seq in pred_ids:
        prev = None
        chars = []
        for token_id in seq.cpu().tolist():
            if token_id != blank_id and token_id != prev:
                char = idx_to_char.get(token_id, "")
                if char not in ["<pad>", "<unk>", "<blank>", ""]:
                    chars.append(char)
            prev = token_id
        decoded.append("".join(chars))

    return decoded


def decode_references(texts, idx_to_char, pad_id, blank_id, unk_id):
    """Convert padded token ids back to reference strings."""
    references = []
    for text_ids in texts:
        text = "".join(
            [
                idx_to_char.get(token_id.item(), "")
                for token_id in text_ids
                if token_id.item() not in [pad_id, blank_id, unk_id]
            ]
        )
        references.append(text)
    return references


def run_inference(model, loader, device, idx_to_char, vocab):
    """Run greedy CTC inference on a loader."""
    model.eval()
    predictions = []
    references = []

    blank_id = vocab["<blank>"]
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test inference"):
            audios = batch["audios"].to(device)
            texts = batch["texts"]

            # Forward
            logits,_ = model(audios, None)

            # Decode predictions
            preds = ctc_decode(logits, idx_to_char=idx_to_char, blank_id=blank_id)

            # Decode references
            refs = decode_references(
                texts,
                idx_to_char=idx_to_char,
                pad_id=pad_id,
                blank_id=blank_id,
                unk_id=unk_id,
            )

            predictions.extend(preds)
            references.extend(refs)

    return predictions, references 

def save_results_json(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_predictions_csv(references, predictions, path):
    df = pd.DataFrame({
        "reference": references,
        "prediction": predictions,
    })
    df.to_csv(path, index=False)