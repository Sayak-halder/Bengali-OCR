import argparse
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import BengaliWordDataset
from models.model import BengaliResNetOCR, BengaliViTOCR, BengaliHybridOCR
from utils.utils import collate_fn, decode_predictions, calculate_cer
from configs.config import Config
from data.augmentation import BengaliAugmentation

def load_model(model_path, model_type, num_chars):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "resnet":
        model = BengaliResNetOCR(num_chars)
    elif model_type == "vit":
        model = BengaliViTOCR(num_chars)
    elif model_type == "hybrid":
        model = BengaliHybridOCR(num_chars)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def visualize_predictions(images, predictions, truths, save_dir="predictions"):
    os.makedirs(save_dir, exist_ok=True)

    denormalize = transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5])
    
    for i, (img, pred, truth) in enumerate(zip(images, predictions, truths)):
        img = denormalize(img).squeeze().cpu().numpy()
        img = (img * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {pred}\nTruth: {truth}", fontsize=12)
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, f"pred_{i}.png"), bbox_inches='tight')
        plt.close()

def evaluate(model_path, data_path, model_type, max_samples=None, visualize=True):
    num_chars = len(Config.BENGALI_CHARS)
    model, device = load_model(model_path, model_type, num_chars)
    print(f"Loaded {model_type} model from {model_path}")

    transform = BengaliAugmentation()
    dataset = BengaliWordDataset(data_path, transform, max_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS
    )

    all_preds = []
    all_truths = []
    all_images = []
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            preds = decode_predictions(outputs.cpu(), Config.IDX_TO_CHAR)
            truths = [''.join([Config.IDX_TO_CHAR[idx.item()] for idx in labels[i][:label_lengths[i]]]) 
                     for i in range(labels.size(0))]
            
            all_preds.extend(preds)
            all_truths.extend(truths)
            all_images.extend(images.cpu())
    
    cer = calculate_cer(all_preds, all_truths)
    print(f"\nCharacter Error Rate (CER): {cer:.4f}")
    
    correct = sum(1 for p, t in zip(all_preds, all_truths) if p == t)
    accuracy = correct / len(all_preds)
    print(f"Word Accuracy: {accuracy:.4f}")
    
    print("\nSample Predictions:")
    for i in range(min(5, len(all_preds))):
        print(f"  Truth: {all_truths[i]}")
        print(f"  Pred:  {all_preds[i]}")
        print()

    if visualize:
        visualize_predictions(all_images, all_preds, all_truths)
        print(f"Saved visualizations to 'predictions' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bengali OCR models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to evaluation data directory")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["resnet", "vit", "hybrid"],
                        help="Type of model architecture")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable prediction visualizations")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        model_type=args.model_type,
        max_samples=args.max_samples,
        visualize=not args.no_visualize,
    )