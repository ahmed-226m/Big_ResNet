
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
try:
    from model.SupCon import create_classifier, VertebraDataset
except ImportError:
    # If running from different directory
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.SupCon import create_classifier, VertebraDataset

def train(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset & Split
    print(f"Loading dataset from {args.input_path}")
    full_dataset = VertebraDataset(args.input_path, args.json_path)
    
    # Simple split: 80% train, 20% val
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty. Check paths.")
        
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    print(f"Data Split: {train_size} Train, {val_size} Val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model
    print(f"Initializing model (Pretrained: {args.pretrained_path})")
    model = create_classifier(pretrained_path=args.pretrained_path, device=device, num_classes=args.num_classes)
    
    # Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # 4. Optimizer & Loss
    # We use a lower LR for the pretrained encoder and higher for the new head (optional, but good for transfer learning)
    # For simplicity, standard Adam is fine for now. weight_decay helps regularization.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Weighted Loss if classes are imbalanced (Optional optimization)
    # Initialize with equal weight for now
    criterion = nn.CrossEntropyLoss()

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 5. Training Loop
    best_f1 = 0.0
    start_time = time.time()
    
    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Calculate Train Metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='binary') # binary for 2 classes
        avg_train_loss = train_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        print("Running validation...")
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, preds = torch.max(output, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Calculate Val Metrics
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        val_precision = precision_score(val_targets, val_preds, average='binary', zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average='binary', zero_division=0)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss) # verbose removed

        # Logging
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f}")
        
        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.output_dir, "best_model.pth")
            
            # Save clean state_dict (unwrap DataParallel)
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
            print(f"  -> Model saved to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='Path to dataset root (containing CT folder)')
    parser.add_argument('--json-path', type=str, required=True, help='Path to vertebra_data.json')
    parser.add_argument('--pretrained-path', type=str, default='kinetics', help="kinetics or path to .pth")
    parser.add_argument('--output-dir', type=str, default='./', help='Where to save best_model.pth')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8) # Lower if OOM
    parser.add_argument('--learning-rate', type=float, default=1e-4) # Standard for transfer learning
    parser.add_argument('--num-classes', type=int, default=2)

    args = parser.parse_args()
    
    # Check output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train(args)
