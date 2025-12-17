"""
Grad-CAM++ Heatmap Generation for HealthiVert-GAN.

This script generates attention heatmaps using Grad-CAM++ from a trained
fracture classifier. The heatmaps guide the HealthiVert-GAN generator to
focus on healthy vertebral regions via the HealthiVert-Guided Attention Module (HGAM).

Usage:
    python Attention/grad_CAM_3d_sagittal.py \\
        --dataroot ./datasets/straightened \\
        --classifier_path ./checkpoints/fracture_classifier/best_model.pth \\
        --output_dir ./Attention/heatmap

Output:
    NIfTI files (.nii.gz) with the same shape as input CT volumes (256, 256, 64)
    Values in range [0, 1] where:
        - High values (close to 1) = healthy regions (classifier confident)
        - Low values (close to 0) = uncertain/fractured regions
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.fracture_classifier import VertebraClassifier, create_classifier
except ImportError:
    # If running from different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.fracture_classifier import VertebraClassifier, create_classifier


class GradCAMPlusPlus3D:
    """
    Grad-CAM++ implementation for 3D CNNs.
    
    Grad-CAM++ improves upon Grad-CAM by using weighted combination of positive
    partial derivatives of the feature maps, providing better localization
    especially when multiple instances of the target class are present.
    
    Reference:
        Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual 
        Explanations for Deep Convolutional Networks", WACV 2018
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The classifier model
            target_layer: The layer to compute Grad-CAM++ on
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input volume (B, C, D, H, W)
            target_class: Class index for gradient computation (None = predicted class)
        
        Returns:
            Heatmap as numpy array with same spatial dimensions as input
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # (B, C, D', H', W')
        gradients = self.gradients      # (B, C, D', H', W')
        
        # Grad-CAM++ weights
        # alpha = grad^2 / (2 * grad^2 + sum(A * grad^3))
        grad_power_2 = gradients ** 2
        grad_power_3 = gradients ** 3
        sum_activations = torch.sum(activations, dim=(2, 3, 4), keepdim=True)
        
        alpha_num = grad_power_2
        alpha_denom = 2 * grad_power_2 + sum_activations * grad_power_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # ReLU on gradients
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        return cam, target_class


def resize_cam_to_volume(cam, target_shape):
    """
    Resize CAM to match the original volume shape.
    
    Args:
        cam: CAM array from Grad-CAM++ (smaller than original)
        target_shape: Target shape (H, W, D) = (256, 256, 64)
    
    Returns:
        Resized CAM matching target_shape
    """
    # CAM comes as (D', H', W'), target is (H, W, D)
    # Need to handle the dimension ordering
    
    if cam.shape == target_shape:
        return cam
    
    # Calculate zoom factors
    # Input CAM is typically (D', H', W') from 3D conv output
    # Need to reshape to (H, W, D) for NIfTI saving
    
    # First, ensure we have the right ordering
    # CAM from model: (D', H', W') where D'=depth after convs
    # Target for NIfTI: (H, W, D) = (256, 256, 64)
    
    target_h, target_w, target_d = target_shape
    current_shape = cam.shape
    
    if len(current_shape) == 3:
        # Assume CAM is (D', H', W')
        # Resize to (target_d, target_h, target_w) then permute
        zoom_factors = (target_d / current_shape[0],
                       target_h / current_shape[1],
                       target_w / current_shape[2])
        cam_resized = zoom(cam, zoom_factors, order=1)
        # Permute from (D, H, W) to (H, W, D)
        cam_resized = cam_resized.transpose(1, 2, 0)
    else:
        # Fallback: direct zoom
        zoom_factors = np.array(target_shape) / np.array(current_shape)
        cam_resized = zoom(cam, zoom_factors, order=1)
    
    return cam_resized


def generate_heatmaps(args):
    """Main function to generate Grad-CAM++ heatmaps for all vertebrae."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load classifier
    print(f"Loading classifier from: {args.classifier_path}")
    
    # Create model with same architecture as training
    model = VertebraClassifier(in_channels=1, num_classes=2, use_se=True)
    
    # Add dropout layer (same as training script)
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 2)
        )
    
    # Load trained weights
    state_dict = torch.load(args.classifier_path, map_location=device)
    
    # Handle DataParallel wrapper if present
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Get target layer for Grad-CAM++
    target_layer = model.get_gradcam_target_layer()
    print(f"Target layer for Grad-CAM++: {target_layer.__class__.__name__}")
    
    # Initialize Grad-CAM++
    gradcam = GradCAMPlusPlus3D(model, target_layer)
    
    # Get list of CT files
    ct_folder = os.path.join(args.dataroot, 'CT')
    if not os.path.exists(ct_folder):
        # Try alternative path
        ct_folder = args.dataroot if os.path.exists(args.dataroot) else None
        if ct_folder is None:
            raise ValueError(f"CT folder not found at {args.dataroot}")
    
    ct_files = sorted([f for f in os.listdir(ct_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    # Filter by JSON if provided
    if args.json_path and os.path.exists(args.json_path):
        import json
        with open(args.json_path, 'r') as f:
            json_data = json.load(f)
        
        labels_map = {}
        for k, v in json_data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    labels_map[sub_k] = sub_v
            else:
                labels_map[k] = v
        
        # Only process files that have labels
        ct_files_filtered = []
        for f in ct_files:
            base = f.replace('.nii.gz', '').replace('.nii', '')
            if base in labels_map:
                ct_files_filtered.append(f)
        ct_files = ct_files_filtered
        print(f"Found {len(ct_files)} CT volumes with labels to process")
    else:
        print(f"Found {len(ct_files)} CT volumes to process")
    
    # Process each vertebra
    for file_name in tqdm(ct_files, desc='Generating heatmaps'):
        vertebra_id = file_name.replace('.nii.gz', '')
        ct_path = os.path.join(ct_folder, file_name)
        output_path = os.path.join(args.output_dir, file_name)
        
        # Skip if already exists
        if os.path.exists(output_path) and not args.overwrite:
            continue
        
        try:
            # Load CT volume
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata().astype(np.float32)
            
            # Normalize to [0, 1]
            data_min, data_max = ct_data.min(), ct_data.max()
            if data_max - data_min > 0:
                ct_data = (ct_data - data_min) / (data_max - data_min)
            
            # Convert to tensor: (H, W, D) -> (1, 1, D, H, W)
            ct_tensor = torch.from_numpy(ct_data).float()
            
            # Permute to correct format: (D, H, W) for model
            if ct_tensor.ndim == 3:
                ct_tensor = ct_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            
            ct_tensor = ct_tensor.to(device)
            
            # Generate Grad-CAM++ heatmap
            with torch.no_grad():
                # Get prediction first
                output = model(ct_tensor)
                pred_class = torch.argmax(output, dim=1).item()
            
            # Generate heatmap for predicted class
            cam = gradcam.generate(ct_tensor, target_class=pred_class)
            
            # Resize CAM to match original volume shape
            target_shape = ct_data.shape  # (H, W, D)
            cam_resized = resize_cam_to_volume(cam, target_shape)
            
            # Invert for HealthiVert-GAN: high values = healthy regions
            # If predicted as fractured (class 1), attention shows fracture
            # We want to highlight healthy regions, so invert
            if pred_class == 1:
                cam_resized = 1.0 - cam_resized
            
            # Save as NIfTI with same affine as original
            cam_nifti = nib.Nifti1Image(cam_resized.astype(np.float32), ct_img.affine)
            nib.save(cam_nifti, output_path)
            
        except Exception as e:
            print(f"Error processing {vertebra_id}: {str(e)}")
            continue
    
    print(f"\nHeatmap generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Total files generated: {len(os.listdir(args.output_dir))}")


def verify_heatmaps(output_dir, sample_count=5):
    """Verify generated heatmaps have correct properties."""
    print("\n" + "="*50)
    print("Heatmap Verification")
    print("="*50)
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')][:sample_count]
    
    all_valid = True
    for file_name in files:
        path = os.path.join(output_dir, file_name)
        cam = nib.load(path).get_fdata()
        
        shape_ok = cam.shape == (256, 256, 64)
        range_ok = 0 <= cam.min() and cam.max() <= 1
        dtype_ok = cam.dtype in [np.float32, np.float64]
        
        status = "✓" if (shape_ok and range_ok and dtype_ok) else "✗"
        print(f"{status} {file_name}: shape={cam.shape}, "
              f"range=[{cam.min():.3f}, {cam.max():.3f}], dtype={cam.dtype}")
        
        if not (shape_ok and range_ok and dtype_ok):
            all_valid = False
    
    if all_valid:
        print("\n✓ All sampled heatmaps are valid!")
    else:
        print("\n✗ Some heatmaps have issues - check the output above")
    
    return all_valid


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM++ Heatmaps')
    
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to straightened dataset (contains CT/ folder)')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='Path to trained classifier weights (.pth)')
    parser.add_argument('--json_path', type=str, default=None,
                        help='Path to JSON labels (optional, filters to labeled samples only)')
    parser.add_argument('--output_dir', type=str, default='./heatmap',
                        help='Output directory for heatmaps')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing heatmaps')
    parser.add_argument('--verify', action='store_true',
                        help='Verify heatmaps after generation')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_heatmaps(args)
    
    if args.verify:
        verify_heatmaps(args.output_dir)


# =============================================================================
# Kaggle Notebook Usage Example
# =============================================================================
"""
# Kaggle Notebook Cells:

# Cell 1: Setup
import sys
sys.path.insert(0, '/kaggle/input/healthivert-gan')
from Attention.grad_CAM_3d_sagittal import *

# Cell 2: Generate heatmaps
class Args:
    dataroot = '/kaggle/working/straightened'
    classifier_path = '/kaggle/working/checkpoints/fracture_classifier/best_model.pth'
    output_dir = '/kaggle/working/Attention/heatmap'
    overwrite = False
    verify = True

generate_heatmaps(Args())

# Cell 3: Verify (optional)
verify_heatmaps('/kaggle/working/Attention/heatmap', sample_count=10)
"""
