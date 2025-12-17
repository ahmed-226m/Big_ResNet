import os
import json
import numpy as np
from collections import Counter

def analyze_dataset(input_path, json_path):
    """Analyze dataset distribution and provide recommendations."""
    
    ct_dir = os.path.join(input_path, 'CT')
    
    # Load JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Flatten JSON structure
    labels_map = {}
    for k, v in json_data.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                labels_map[sub_k] = sub_v
        else:
            labels_map[k] = v
    
    # Get files in dataset
    if not os.path.exists(ct_dir):
        print(f"Error: CT directory not found at {ct_dir}")
        return
    
    files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    # Match files with labels
    matched_samples = []
    unmatched_files = []
    
    for f in files:
        base = f.replace('.nii.gz', '').replace('.nii', '')
        if base in labels_map:
            label = labels_map[base]
            binary_label = 0 if label == 0 else 1
            matched_samples.append((base, label, binary_label))
        else:
            unmatched_files.append(base)
    
    # Analyze labels
    all_labels = [s[1] for s in matched_samples]
    binary_labels = [s[2] for s in matched_samples]
    
    label_counter = Counter(all_labels)
    binary_counter = Counter(binary_labels)
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    print("\n1. DATASET SIZE:")
    print(f"   Total files in CT folder: {len(files)}")
    print(f"   Matched samples (file + label): {len(matched_samples)}")
    print(f"   Unmatched files: {len(unmatched_files)}")
    print(f"   Labels in JSON without files: {len([k for k in labels_map.keys() if k not in [s[0] for s in matched_samples]])}")
    
    print("\n2. GENANT GRADE DISTRIBUTION:")
    for grade in sorted(label_counter.keys()):
        count = label_counter[grade]
        pct = 100 * count / len(matched_samples)
        print(f"   Grade {grade}: {count:4d} samples ({pct:5.1f}%)")
    
    print("\n3. BINARY CLASSIFICATION DISTRIBUTION:")
    for bin_label in sorted(binary_counter.keys()):
        count = binary_counter[bin_label]
        pct = 100 * count / len(matched_samples)
        label_name = "Normal (grade 0)" if bin_label == 0 else "Fractured (grade 1-3)"
        print(f"   Class {bin_label} ({label_name}): {count:4d} samples ({pct:5.1f}%)")
    
    print("\n4. RECOMMENDED SPLIT SIZES (80/20 train/val):")
    train_size = int(0.8 * len(matched_samples))
    val_size = len(matched_samples) - train_size
    
    class_1_ratio = binary_counter[1] / len(matched_samples)
    expected_val_class_1 = int(val_size * class_1_ratio)
    
    print(f"   Training: {train_size} samples (~{int(train_size * (1-class_1_ratio))} class 0, ~{int(train_size * class_1_ratio)} class 1)")
    print(f"   Validation: {val_size} samples (~{val_size - expected_val_class_1} class 0, ~{expected_val_class_1} class 1)")
    
    print("\n5. POTENTIAL ISSUES:")
    issues = []
    
    # Check class imbalance
    imbalance_ratio = binary_counter[0] / binary_counter[1]
    if imbalance_ratio > 3.0:
        issues.append(f"   ⚠️  Severe class imbalance (ratio {imbalance_ratio:.1f}:1)")
        issues.append(f"      → Using class weights is essential")
    
    # Check validation size
    if expected_val_class_1 < 30:
        issues.append(f"   ⚠️  Very few fractured samples in validation ({expected_val_class_1})")
        issues.append(f"      → Metrics will be unstable, consider k-fold cross-validation")
    
    # Check total dataset size
    if len(matched_samples) < 500:
        issues.append(f"   ⚠️  Small dataset ({len(matched_samples)} samples)")
        issues.append(f"      → Risk of overfitting, use strong regularization")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("   ✓ No major issues detected")
    
    print("\n6. RECOMMENDATIONS:")
    print("   1. Use stratified split to maintain class balance")
    print("   2. Add dropout (0.3-0.5) and weight decay (1e-4)")
    print("   3. Use data augmentation if possible")
    print("   4. Consider k-fold cross-validation for stable metrics")
    print("   5. Use early stopping based on validation F1 score")
    print("   6. Use learning rate scheduling")
    
    print("\n7. SAMPLE DISTRIBUTION BY PATIENT:")
    patient_counts = {}
    for base, label, bin_label in matched_samples:
        patient = base.split('_')[0]
        if patient not in patient_counts:
            patient_counts[patient] = {'total': 0, 'fractured': 0}
        patient_counts[patient]['total'] += 1
        if bin_label == 1:
            patient_counts[patient]['fractured'] += 1
    
    print(f"   Total patients: {len(patient_counts)}")
    fractured_patients = sum(1 for p in patient_counts.values() if p['fractured'] > 0)
    print(f"   Patients with fractures: {fractured_patients}")
    print(f"   Patients without fractures: {len(patient_counts) - fractured_patients}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze vertebra dataset')
    parser.add_argument('--input-path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--json-path', type=str, required=True, help='Path to JSON labels')
    
    args = parser.parse_args()
    
    analyze_dataset(args.input_path, args.json_path)
