import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm

from models.model import BengaliResNetOCR, BengaliViTOCR, BengaliHybridOCR
from utils.utils import collate_fn, decode_predictions, calculate_cer
from configs.config import Config
from data.dataset import BengaliWordDataset
from data.augmentation import BengaliAugmentation


def train_model(model, train_loader, val_loader, device, model_type):
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CTCLoss(blank=0)
    
    best_cer = float('inf')
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for images, labels, label_lengths in tqdm(val_loader):
                images = images.to(device)
                outputs = model(images)
                preds = decode_predictions(outputs.cpu(), Config.IDX_TO_CHAR)
                truths = [''.join([Config.IDX_TO_CHAR[idx.item()] for idx in labels[i][:label_lengths[i]]]) for i in range(labels.size(0))]
                val_preds.extend(preds)
                val_truths.extend(truths)
        
        val_cer = calculate_cer(val_preds, val_truths)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val CER: {val_cer:.4f}")
        
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), f"best_{model_type}_model.pth")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bengali OCR models")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["resnet", "vit", "hybrid"],
                        help="Type of model to train")
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to training data directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use for training")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = BengaliAugmentation()
    dataset = BengaliWordDataset(args.data_path, transform, args.max_samples)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_set, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn, 
        num_workers=Config.NUM_WORKERS
    )

    num_chars = len(Config.BENGALI_CHARS)
    if args.model == "resnet":
        model = BengaliResNetOCR(num_chars)
    elif args.model == "vit":
        model = BengaliViTOCR(num_chars)
    elif args.model == "hybrid":
        model = BengaliHybridOCR(num_chars)
    
    model = model.to(device)
    print(f"Training {args.model} model with {len(dataset)} samples")
    print(f"Batch size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}, LR: {Config.LEARNING_RATE}")
    
    trained_model = train_model(model, train_loader, val_loader, device, args.model)
    print(f"Training completed! Best model saved as best_{args.model}_model.pth")