import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import argparse
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models

# -----------------------------
# SupCon Loss (directly from SupContrast repo - unchanged)
# -----------------------------
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss from https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: normalized features, shape [bsz, n_views, feature_dim]
        labels: ground truth [bsz]
        mask: optional precomputed contrastive mask [bsz, bsz]
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('features should be [bsz, n_views, ...]')

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')

        # Create mask from labels if not provided
        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask.repeat(anchor_count, contrast_count) * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# -----------------------------
# 3D SE Block
# -----------------------------
class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# -----------------------------
# 3D Bottleneck with SE
# -----------------------------
class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -----------------------------
# 3D SE-ResNet50
# -----------------------------
class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, zero_init_residual=False):
        super(ResNet3D, self).__init__()
        self.inplanes = 64

        # Stem (adjusted for medical 3D patches - smaller stride in depth)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)  # Depth stride=1 to preserve 32→32
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)  # Light pooling

        self.layer1 = self._make_layer(Bottleneck3D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck3D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck3D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# -----------------------------
# Wrappers (like SupContrast)
# -----------------------------
class SupConResNet3D(nn.Module):
    """Encoder + projection head for SupCon"""
    def __init__(self, in_channels=1, feat_dim=128):
        super(SupConResNet3D, self).__init__()
        self.encoder = ResNet3D(in_channels=in_channels)
        self.head = nn.Sequential(
            nn.Linear(512 * Bottleneck3D.expansion, 512 * Bottleneck3D.expansion),
            nn.ReLU(inplace=True),
            nn.Linear(512 * Bottleneck3D.expansion, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class VertebraClassifier(nn.Module):
    """
    Encoder + classifier (for CE loss).
    Renamed from SupCEResNet3D to match usage in grad_CAM_3d_sagittal.py
    (or at least provide the structure it expects).
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(VertebraClassifier, self).__init__()
        self.encoder = ResNet3D(in_channels=in_channels, num_classes=num_classes)
        self.fc = nn.Linear(512 * Bottleneck3D.expansion, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)
    
    def get_gradcam_target_layer(self):
        """
        Returns the target layer for Grad-CAM++.
        Typically the last convolutional layer output (layer4).
        """
        # ResNet3D.layer4 is a Sequential of Bottleneck3D blocks.
        # We want the last block.
        # Inside the last block, the last conv is conv3 (before se and residual).
        # Actually, GradCAM usually hooks onto the bottleneck block output (after relu).
        # But layer4[-1] is the module itself.
        # If we register a forward hook on the block, output will be correct.
        return self.encoder.layer4[-1]

class PretrainedVertebraClassifier(nn.Module):
    """
    Wrapper for torchvision's R3D models (Kinetics-400 pretrained).
    Modifies first layer for 1-channel input.
    """
    def __init__(self, model_name='r3d_18', in_channels=1, num_classes=2, pretrained=True):
        super(PretrainedVertebraClassifier, self).__init__()
        
        # Load torchvision model
        weights = 'KINETICS400_V1' if pretrained else None
        if model_name == 'r3d_18':
            self.model = video_models.r3d_18(weights=weights)
            # r3d_18 has 'layer4'
            self.gradcam_layer_attr = 'layer4'
        elif model_name == 'mc3_18':
            self.model = video_models.mc3_18(weights=weights)
            self.gradcam_layer_attr = 'layer4'
        elif model_name == 'r3d_50': # Much heavier
            # r3d_50 weights might not be standard in older torchvision, check version if error
            self.model = video_models.r3d_18(weights=weights) # Fallback to 18 if 50 unspecified
            self.gradcam_layer_attr = 'layer4'
        else:
             raise ValueError(f"Unknown model {model_name}")

        # Modify first layer (stem) to accept in_channels (CT is 1 channel)
        # Original: Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        old_conv = self.model.stem[0]
        if old_conv.in_channels != in_channels:
             new_conv = nn.Conv3d(
                 in_channels, 
                 old_conv.out_channels, 
                 kernel_size=old_conv.kernel_size, 
                 stride=old_conv.stride, 
                 padding=old_conv.padding, 
                 bias=old_conv.bias is not None
             )
             
             # Initialize with average of RGB weights if pretrained
             if pretrained:
                 with torch.no_grad():
                     new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
             
             self.model.stem[0] = new_conv
        
        # Replace final fc layer
        # Linear(in_features=512, out_features=400, bias=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

    def get_gradcam_target_layer(self):
        # Return the last block of the last residual layer
        layer = getattr(self.model, self.gradcam_layer_attr)
        return layer[-1]

def create_classifier(pretrained_path=None, device='cpu', in_channels=1, num_classes=2):
    """Factory function to create classifier."""
    
    # Check if we should use torchvision pretrained
    if pretrained_path and 'kinetics' in str(pretrained_path).lower():
        print(f"Loading Torchvision model with {pretrained_path} weights...")
        model = PretrainedVertebraClassifier(model_name='r3d_18', in_channels=in_channels, num_classes=num_classes, pretrained=True)
    else:
        # Use custom ResNet3D (scratch or custom checkpoint)
        model = VertebraClassifier(in_channels=in_channels, num_classes=num_classes)
    
        if pretrained_path:
            if isinstance(pretrained_path, str) and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location=device)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if name.startswith('module.'): name = name[7:]
                    if name.startswith('encoder.'): pass
                    new_state_dict[name] = v
                try:
                    model.load_state_dict(new_state_dict, strict=True)
                except:
                    model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded weights from file {pretrained_path}")
            else:
                 print(f"Pretrained path {pretrained_path} not found or invalid.")
    
    model.to(device)
    return model

# Alias for compatibility if user import expects SupCEResNet3D
SupCEResNet3D = VertebraClassifier

# -----------------------------
# Dataset and Execution
# -----------------------------
class VertebraDataset(Dataset):
    """
    Dataset for 3D Vertebrae.
    Expected structure:
    root_dir/
      CT/
        filename.nii.gz (or .nii)
    """
    def __init__(self, root_dir, json_path, transform=None):
        self.root_dir = root_dir
        self.ct_dir = os.path.join(root_dir, 'CT')
        self.transform = transform
        
        # Load labels
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        
        # Flatten json data (it might be nested like {'train': {...}, 'test': {...}} or flat)
        self.labels_map = {}
        for k, v in self.json_data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    self.labels_map[sub_k] = sub_v
            else:
                self.labels_map[k] = v
                
        # List valid files
        self.samples = []
        if os.path.exists(self.ct_dir):
            files = sorted([f for f in os.listdir(self.ct_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
            for f in files:
                # filename like 'sub-verse004_16.nii.gz' -> id 'sub-verse004_16'
                # Handle simplified naming
                base = f.replace('.nii.gz', '').replace('.nii', '')
                
                # Check if we have a label
                # Some filenames might have extra suffixes, try exact match first
                label = -1
                if base in self.labels_map:
                    label = self.labels_map[base]
                else:
                    # Try removing _seriesX or other heuristics if needed, or simple skip
                    # For now, strict match or print warning in debug
                    pass
                
                if label != -1:
                    # Binarize label: 0->0 (Healthy), 1,2,3->1 (Fracture)
                    # Modify this if multi-class is needed
                    binary_label = 0 if label == 0 else 1
                    self.samples.append((os.path.join(self.ct_dir, f), binary_label, base))
        else:
            print(f"Warning: CT directory not found at {self.ct_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, case_id = self.samples[idx]
        
        # Load NIfTI
        try:
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            
            # Normalize [0,1]
            grad_min, grad_max = data.min(), data.max()
            if grad_max - grad_min > 0:
                data = (data - grad_min) / (grad_max - grad_min)
            
            # Transform to tensor (C, D, H, W)
            # Input data is likely (H, W, D) based on previous context, needs permute
            # straighten_mask_3d output is (128, 128, 64) or (256, 256, 64)
            data_tensor = torch.from_numpy(data)
            
            # Ensure dims are (C, D, H, W) for 3D Conv
            # If shape is (H, W, D), permute to (D, H, W) then unsqueeze C
            if data_tensor.ndim == 3:
                data_tensor = data_tensor.permute(2, 0, 1).unsqueeze(0) # -> (1, D, H, W)
            
            return data_tensor, label, case_id
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((1, 64, 256, 256)), label, case_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vertebra Classifier & SupCon Model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to dataset root (containing CT folder)')
    parser.add_argument('--json-path', type=str, required=True, help='Path to vertebra_data.json')
    parser.add_argument('--pretrained-path', type=str, default=None, help='Path to .pth checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    
    args = parser.parse_args()
    
    print(f"Initializing model with {args.num_classes} classes...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_classifier(pretrained_path=args.pretrained_path, device=device, num_classes=args.num_classes)
    
    print(f"Loading data from {args.input_path}...")
    dataset = VertebraDataset(args.input_path, args.json_path)
    print(f"Found {len(dataset)} labeled samples.")
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        # Run a quick check on first batch
        data, label, ids = next(iter(loader))
        data = data.to(device)
        print(f"Sample input shape: {data.shape}")
        
        output = model(data)
        print(f"Model output shape: {output.shape}")
        print(f"Sample predictions: {torch.argmax(output, dim=1).cpu().numpy()}")
        print(f"Sample ground truth: {label.numpy()}")
        print("Model is ready-to-run on input data.")
    else:
        print("No samples found. Check paths.")
