# Save this as Attention/train_classifier_custom.py
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nibabel as nib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import the OLD classifier
try:
    from model.fracture_classifier import VertebraClassifier
except ImportError:
    # If running from different directory
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.fracture_classifier import VertebraClassifier
# We need to redefine Dataset because it's not in fracture_classifier.py
class VertebraDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None):
        self.root_dir = root_dir
        self.ct_dir = os.path.join(root_dir, 'CT')
        self.transform = transform
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        
        # Flatten the nested JSON structure
        self.labels_map = {}
        for k, v in self.json_data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    self.labels_map[sub_k] = sub_v
            else:
                self.labels_map[k] = v
        
        self.samples = []
        
        if os.path.exists(self.ct_dir):
            # Get all files in CT directory
            files = sorted([f for f in os.listdir(self.ct_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
            
            # Create a set of base names from files
            file_base_names = set()
            for f in files:
                base = f.replace('.nii.gz', '').replace('.nii', '')
                file_base_names.add(base)
                file_path = os.path.join(self.ct_dir, f)
                
                # Only add samples that have both file AND label
                if base in self.labels_map and os.path.isfile(file_path):
                    label = self.labels_map[base]
                    binary_label = 0 if label == 0 else 1
                    self.samples.append((file_path, binary_label))
            
            # Calculate actual mismatches
            skipped_count = len(file_base_names - set(self.labels_map.keys()))
            json_only_count = len(set(self.labels_map.keys()) - file_base_names)
        
        # Print dataset statistics
        class_0_count = sum(1 for _, label in self.samples if label == 0)
        class_1_count = sum(1 for _, label in self.samples if label == 1)
        
        print("\n" + "="*60)
        print("Dataset Statistics:")
        print(f"  Total samples loaded: {len(self.samples)}")
        print(f"  Class 0 (Normal): {class_0_count} ({100*class_0_count/len(self.samples):.1f}%)" if len(self.samples) > 0 else "  No samples")
        print(f"  Class 1 (Fractured): {class_1_count} ({100*class_1_count/len(self.samples):.1f}%)" if len(self.samples) > 0 else "")
        print(f"  Files in dataset without JSON labels: {skipped_count}")
        print(f"  Labels in JSON without dataset files: {json_only_count}")
        print("="*60 + "\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            grad_min, grad_max = data.min(), data.max()
            if grad_max - grad_min > 0:
                data = (data - grad_min) / (grad_max - grad_min)
            data_tensor = torch.from_numpy(data)
            # Permute to (C=1, D, H, W)
            if data_tensor.ndim == 3:
                data_tensor = data_tensor.permute(2, 0, 1).unsqueeze(0)
            return data_tensor, label, "id"
        except:
            return torch.zeros((1, 64, 256, 256)), label, "error"

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VertebraDataset(args.input_path, args.json_path)
    
    # Stratified split to maintain class balance
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    class_0_indices = [i for i, label in enumerate(all_labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(all_labels) if label == 1]
    
    np.random.seed(42)
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    val_size_0 = int(0.2 * len(class_0_indices))
    val_size_1 = int(0.2 * len(class_1_indices))
    
    val_indices = class_0_indices[:val_size_0] + class_1_indices[:val_size_1]
    train_indices = class_0_indices[val_size_0:] + class_1_indices[val_size_1:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"\nStratified Split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize Custom Model (Random Weights)
    model = VertebraClassifier(in_channels=1, num_classes=args.num_classes, use_se=True)
    
    # Add dropout for regularization
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, args.num_classes)
        )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Calculate class weights for imbalanced dataset
    class_0_count = 0
    class_1_count = 0
    for i in range(len(train_dataset)):
        _, label, _ = train_dataset[i]
        if label == 0:
            class_0_count += 1
        else:
            class_1_count += 1
    
    total = len(train_dataset)
    
    # Inverse frequency weighting
    weight_0 = total / (2.0 * class_0_count) if class_0_count > 0 else 1.0
    weight_1 = total / (2.0 * class_1_count) if class_1_count > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_f1 = 0.0
    patience_counter = 0
    patience_limit = 7
    
    print(f"Class weights: Class 0 = {weight_0:.4f}, Class 1 = {weight_1:.4f}")
    print(f"Weight decay: 1e-4, Dropout: 0.4, Early stopping patience: {patience_limit}\n")

    for epoch in range(args.epochs):
        model.train()
        train_preds, train_targets = [], []
        
        for i, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        # Metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='binary')
        train_precision = precision_score(train_targets, train_preds, average='binary', zero_division=0)
        train_recall = recall_score(train_targets, train_preds, average='binary', zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='binary', zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average='binary', zero_division=0)
        
        print("-" * 50)
        print(f"Epoch {epoch+1}/{args.epochs} Report:")
        print(f"Train | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f}")
        print(f"Val   | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | Loss: {val_loss/len(val_loader):.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.6f}")
        print("-" * 50)

        # Save best model and early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                       os.path.join(args.output_dir, "best_model_custom.pth"))
            print(f"✓ Saved Best Model (F1: {best_f1:.4f})\n")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience_limit})\n")
            
            if patience_counter >= patience_limit:
                print(f"\n{'='*50}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation F1: {best_f1:.4f}")
                print(f"{'='*50}")
                break
    
    print(f"\nTraining completed. Best validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--json-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()
    train(args)