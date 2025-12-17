import torch
import torch.nn as nn
import torch.nn.functional as F

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

def create_classifier(pretrained_path=None, device='cpu', in_channels=1, num_classes=2):
    """Factory function to create and load classifier."""
    model = VertebraClassifier(in_channels=in_channels, num_classes=num_classes)
    model.to(device)
    
    if pretrained_path:
        if isinstance(pretrained_path, str) and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            # Handle potential DataParallel wrapping or SupCon prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]
                if name.startswith('encoder.'): # If loading from SupCon-style but we are same class
                     pass # Matches our structure
                new_state_dict[name] = v
            
            # If strict=False, we can load what matches.
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: strict loading failed, trying strict=False. Error: {e}")
                model.load_state_dict(new_state_dict, strict=False)
        else:
             print(f"Pretrained path {pretrained_path} not found or invalid.")
    
    return model

# Alias for compatibility if user import expects SupCEResNet3D
SupCEResNet3D = VertebraClassifier