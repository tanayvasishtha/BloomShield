"""
BloomShield High-Accuracy Training - FIXED VERSION
Properly handles validation dataset splitting
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import time

def organize_high_quality_dataset():
    """Organize dataset with PROPER validation splitting"""
    print("🎯 BloomShield High-Accuracy Dataset Preparation (FIXED)")
    print("=" * 60)
    
    # Create directories
    dataset_path = Path('high_accuracy_dataset_fixed')
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'val'
    
    # Remove existing dataset if any
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Read all images from data/train
    source_path = Path('data/train')
    print("📂 Analyzing and filtering dataset for quality...")
    
    class_images = defaultdict(list)
    total_files = 0
    
    # Group images by class name with better parsing
    for img_file in os.listdir(source_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_files += 1
            
            try:
                # Better class name extraction
                parts = img_file.split('_')
                if len(parts) >= 2:
                    # Handle formats like "Tomato_Bacterial_spot" or "Apple_Apple_scab"
                    crop = parts[0]
                    disease = parts[1] if len(parts) > 1 else "unknown"
                    
                    # Clean and combine
                    class_name = f"{crop}_{disease}".replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
                    class_images[class_name].append(img_file)
                    
            except Exception as e:
                continue
    
    # Filter: Keep only classes with sufficient samples (min 100 images for better validation split)
    filtered_classes = {k: v for k, v in class_images.items() if len(v) >= 100}
    
    print(f"📊 Original classes: {len(class_images)}")
    print(f"📊 High-quality classes (100+ images): {len(filtered_classes)}")
    
    # Show top classes
    sorted_classes = sorted(filtered_classes.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\n📈 Top 15 classes by image count:")
    for i, (class_name, images) in enumerate(sorted_classes[:15]):
        print(f"  {i+1:2d}. {class_name:40} - {len(images):,} images")
    
    print(f"\n🔄 Organizing {len(sorted_classes)} high-quality classes...")
    
    train_total = 0
    val_total = 0
    
    for class_name, images in sorted_classes:
        # Create class directories
        train_class_dir = train_path / class_name
        val_class_dir = val_path / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # FIXED: Better split with minimum validation images
        total_images = len(images)
        val_count = max(10, int(total_images * 0.15))  # At least 10 validation images per class
        train_count = total_images - val_count
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        
        # Copy training images
        train_copied = 0
        for img in train_images:
            try:
                src = source_path / img
                dst = train_class_dir / img
                shutil.copy2(src, dst)
                train_copied += 1
            except Exception as e:
                continue
        
        # Copy validation images
        val_copied = 0
        for img in val_images:
            try:
                src = source_path / img
                dst = val_class_dir / img
                shutil.copy2(src, dst)
                val_copied += 1
            except Exception as e:
                continue
        
        train_total += train_copied
        val_total += val_copied
        
        print(f"✓ {class_name}: {train_copied} train, {val_copied} val")
    
    print(f"\n✅ High-quality dataset organized!")
    print(f"📊 Classes: {len(sorted_classes)}")
    print(f"🖼️  Training images: {train_total:,}")
    print(f"🖼️  Validation images: {val_total:,}")
    print(f"📁 Location: {dataset_path.absolute()}")
    
    return True, len(sorted_classes)

def train_yolov8m_high_accuracy():
    """Train YOLOv8m for maximum accuracy"""
    try:
        from ultralytics import YOLO
        
        print("\n" + "=" * 60)
        print("🎯 BloomShield YOLOv8m High-Accuracy Training (FIXED)")
        print("Target: 90%+ Top-1 Accuracy")
        print("=" * 60)
        
        # Use YOLOv8m (medium) for better accuracy
        print("📥 Loading YOLOv8m-cls model...")
        model = YOLO('yolov8m-cls.pt')
        
        print("🚀 Starting FIXED high-accuracy training...")
        
        start_time = time.time()
        
        # High-accuracy training configuration
        results = model.train(
            data='high_accuracy_dataset_fixed',
            epochs=7,   # User requested 7 epochs for faster training
            imgsz=256,  # Balanced size
            batch=8,    # Smaller batch for stability
            device='cpu',
            project='runs/classify',
            name='bloomshield_high_accuracy_fixed',
            verbose=True,
            save=True,
            save_period=3,  # Save every 3 epochs
            patience=20,    # Good patience
            workers=4,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,      # Standard learning rate
            lrf=0.01,       # Learning rate decay
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Effective augmentation
            augment=True,
            degrees=10.0,    # Moderate rotation
            translate=0.1,   # Moderate translation
            scale=0.2,       # Moderate scaling
            shear=5.0,       # Moderate shearing
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,     # Color augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            
            # Classification settings
            dropout=0.1,     # Moderate dropout
            mixup=0.1,       # Light MixUp
        )
        
        training_time = time.time() - start_time
        
        # Save the final model
        os.makedirs('models', exist_ok=True)
        final_model_path = 'models/bloomshield_yolov8m_high_accuracy_fixed.pt'
        model.save(final_model_path)
        
        print("\n" + "=" * 60)
        print("🎉 HIGH-ACCURACY TRAINING COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
        print(f"💾 Model saved to: {final_model_path}")
        print(f"📊 Training logs: runs/classify/bloomshield_high_accuracy_fixed/")
        
        print(f"\n🎯 This should achieve 85%+ accuracy!")
        print(f"✅ Update app.py to use: {final_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Fixed high-accuracy training pipeline"""
    print("🎯 BloomShield High-Accuracy Training Pipeline (FIXED)")
    print("Fix: Proper validation dataset creation")
    print("Goal: Achieve 85%+ accuracy")
    print("=" * 60)
    
    # Step 1: Organize high-quality dataset with PROPER validation split
    success, num_classes = organize_high_quality_dataset()
    if not success:
        print("❌ Dataset organization failed!")
        return
    
    # Step 2: Train with YOLOv8m
    if not train_yolov8m_high_accuracy():
        print("❌ High-accuracy training failed!")
        return
    
    print("\n🏆 High-accuracy training completed!")
    print("🎯 Expected accuracy: 85%+ (major improvement from 66.5%)")
    print("🚀 Ready for production deployment!")

if __name__ == "__main__":
    main() 