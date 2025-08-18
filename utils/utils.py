import torch
import editdistance

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max(label_lengths)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, (seq, seq_len) in enumerate(zip(labels, label_lengths)):
        padded_labels[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
    return images, padded_labels, torch.tensor(label_lengths, dtype=torch.long)

def decode_predictions(outputs, idx_to_char):
    _, max_indices = torch.max(outputs, 2)
    predictions = []
    for indices in max_indices.permute(1, 0):
        decoded = []
        prev_char = 0
        for idx in indices:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                decoded.append(idx_to_char[idx])
            prev_char = idx
        predictions.append(''.join(decoded))
    return predictions

def calculate_cer(predictions, truths):
    total_errors = sum(editdistance.eval(pred, truth) for pred, truth in zip(predictions, truths))
    total_chars = sum(len(truth) for truth in truths)
    return total_errors / total_chars