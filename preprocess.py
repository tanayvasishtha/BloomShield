"""
BloomShield Dataset Preprocessing for YOLOv8m
Prepares the New Plant Diseases Dataset for training
"""

import os
import shutil
from pathlib import Path
import yaml
from PIL import Image
import random

def create_yolo_structure():
    """Create YOLO dataset directory structure"""
    directories = [
        'datasets/train/images',
        'datasets/train/labels', 
        'datasets/val/images',
        'datasets/val/labels',
        'datasets/test/images',
        'datasets/test/labels'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ YOLO directory structure created")

def get_class_names():
    """Extract class names from dataset structure"""
    train_dir = 'data/train'
    if not os.path.exists(train_dir):
        print("❌ Training data directory not found!")
        return []
    
    # Get class names from directory structure
    class_names = []
    for item in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, item)):
            class_names.append(item)
    
    class_names.sort()  # Ensure consistent ordering
    print(f"✓ Found {len(class_names)} classes: {class_names[:5]}..." if len(class_names) > 5 else f"✓ Found classes: {class_names}")
    return class_names

def convert_to_yolo_format(source_dir, target_images_dir, target_labels_dir, class_names):
    """Convert classification dataset to YOLO detection format"""
    converted_count = 0
    
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        class_id = class_names.index(class_name)
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Verify image is valid
                    img_path = os.path.join(class_dir, img_file)
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    # Copy image to target directory
                    target_img_path = os.path.join(target_images_dir, f"{class_name}_{img_file}")
                    shutil.copy2(img_path, target_img_path)
                    
                    # Create corresponding label file
                    label_file = Path(target_img_path).stem + '.txt'
                    label_path = os.path.join(target_labels_dir, label_file)
                    
                    # For classification, create full-image bounding box
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                    
                    converted_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Skipping {img_file}: {str(e)}")
                    continue
    
    return converted_count

def create_dataset_yaml(class_names):
    """Create YOLO dataset configuration file"""
    dataset_config = {
        'path': str(Path('datasets').absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open('datasets/data.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Dataset config saved with {len(class_names)} classes")
    return 'datasets/data.yaml'

def prepare_dataset():
    """Main preprocessing pipeline"""
    print("🌱 BloomShield Dataset Preprocessing Started")
    print("=" * 50)
    
    # Step 1: Create directory structure
    create_yolo_structure()
    
    # Step 2: Get class names
    class_names = get_class_names()
    if not class_names:
        print("❌ No classes found. Please check your data directory structure.")
        return None
    
    # Step 3: Convert training data
    print("📁 Converting training data...")
    train_count = convert_to_yolo_format(
        'data/train',
        'datasets/train/images',
        'datasets/train/labels',
        class_names
    )
    print(f"✓ Converted {train_count} training images")
    
    # Step 4: Convert validation data
    print("📁 Converting validation data...")
    val_count = convert_to_yolo_format(
        'data/valid',
        'datasets/val/images', 
        'datasets/val/labels',
        class_names
    )
    print(f"✓ Converted {val_count} validation images")
    
    # Step 5: Convert test data (if exists)
    if os.path.exists('data/test'):
        print("📁 Converting test data...")
        test_count = convert_to_yolo_format(
            'data/test',
            'datasets/test/images',
            'datasets/test/labels', 
            class_names
        )
        print(f"✓ Converted {test_count} test images")
    
    # Step 6: Create dataset configuration
    yaml_path = create_dataset_yaml(class_names)
    
    print("=" * 50)
    print("🎉 Dataset preprocessing completed!")
    print(f"📊 Classes: {len(class_names)}")
    print(f"🖼️  Total images: {train_count + val_count}")
    print(f"📄 Config file: {yaml_path}")
    
    return yaml_path

if __name__ == "__main__":
    prepare_dataset() 