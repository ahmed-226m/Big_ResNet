"""
3D Vertebra Fracture Classifier for Grad-CAM++ Generation.

This classifier is used to generate attention heatmaps that guide
HealthiVert-GAN to focus on healthy vertebral regions.

Architecture: 3D ResNet-18 adapted for single-channel CT volumes
Input: Straightened vertebra volume (256, 256, 64)
Output: Binary classification (normal vs fractured)

Paper Reference: HealthiVert-GAN uses Grad-CAM++ from a fracture 
classifier to implement the HealthiVert-Guided Attention Module (HGAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """Basic 3D Residual Block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D volumes."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SEResBlock3D(nn.Module):
    """3D Residual Block with Squeeze-and-Excitation."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels, reduction)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class VertebraClassifier(nn.Module):
    """
    3D SE-ResNet Classifier for Vertebral Fracture Detection.
    
    Binary classification: 
        - Class 0: Normal/Mild (Genant grade 0-1)
        - Class 1: Moderate/Severe (Genant grade 2-3)
    
    The last convolutional layer (layer4) is used for Grad-CAM++ extraction.
    """
    
    def __init__(self, in_channels=1, num_classes=2, use_se=True):
        super(VertebraClassifier, self).__init__()
        
        self.use_se = use_se
        block = SEResBlock3D if use_se else ResBlock3D
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(block, 64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(block, 128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(block, 256, 512, blocks=2, stride=2)  # Target for Grad-CAM++
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: (B, 1, D, H, W) -> typically (B, 1, 64, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Features for Grad-CAM++
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_gradcam_target_layer(self):
        """Returns the target layer for Grad-CAM++ extraction."""
        return self.layer4


def create_classifier(pretrained_path=None, device='cuda'):
    """
    Create and optionally load pretrained classifier.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model on specified device
    """
    model = VertebraClassifier(in_channels=1, num_classes=2, use_se=True)
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    model = VertebraClassifier()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test input: (batch, channels, depth, height, width)
    # Note: depth=64, height=256, width=256 for straightened vertebra volumes
    x = torch.randn(2, 1, 64, 256, 256)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")  # Should be (2, 2)
        print(f"Predictions: {torch.argmax(output, dim=1)}")
    
    # Test Grad-CAM target layer
    target_layer = model.get_gradcam_target_layer()
    print(f"Grad-CAM target layer: {target_layer}")
