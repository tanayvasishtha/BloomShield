"""
BloomShield YOLOv8m Training Script
Trains YOLOv8m model on New Plant Diseases Dataset for crop disease detection
"""

import os
import time
from datetime import datetime
from ultralytics import YOLO
import torch
from preprocess import prepare_dataset

# Training configuration
EPOCHS = 8
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 'cpu'  # Change to 'cuda' if GPU available
MODEL_SIZE = 'yolov8m.pt'  # YOLOv8 medium for balance of speed/accuracy

def setup_training_environment():
    """Setup directories and check requirements"""
    print("🔧 Setting up training environment...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Check if data exists
    if not os.path.exists('data/train'):
        print("❌ Training data not found in 'data/train'")
        print("Please ensure your dataset is extracted to the 'data' directory")
        return False
    
    print("✓ Training environment ready")
    return True

def train_bloomshield_model():
    """Main training function"""
    print("🌱 BloomShield YOLOv8m Training Started")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup environment
    if not setup_training_environment():
        return None
    
    # Prepare dataset
    print("📊 Preparing dataset...")
    dataset_yaml = prepare_dataset()
    if not dataset_yaml:
        print("❌ Dataset preparation failed")
        return None
    
    # Initialize YOLOv8m model
    print(f"🤖 Loading YOLOv8m model ({MODEL_SIZE})...")
    model = YOLO(MODEL_SIZE)
    
    # Display model info
    print(f"📋 Model: YOLOv8m")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🖼️  Image size: {IMAGE_SIZE}")
    print(f"💻 Device: {DEVICE}")
    
    # Start training
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project='runs/train',
            name='bloomshield_yolov8m',
            save=True,
            save_period=2,  # Save checkpoint every 2 epochs
            verbose=True,
            patience=5,  # Early stopping patience
            workers=4,
            optimizer='Adam',
            lr0=0.001,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Box loss gain
            cls=0.5,  # Classification loss gain
            dfl=1.5,  # Distribution Focal Loss gain
            mosaic=1.0,  # Mosaic augmentation probability
            mixup=0.1,  # MixUp augmentation probability
            copy_paste=0.1,  # Copy-paste augmentation probability
            degrees=0.0,  # Rotation degrees
            translate=0.1,  # Translation
            scale=0.5,  # Scaling
            shear=0.0,  # Shearing
            perspective=0.0,  # Perspective
            flipud=0.0,  # Vertical flip probability
            fliplr=0.5,  # Horizontal flip probability
            hsv_h=0.015,  # Hue augmentation
            hsv_s=0.7,  # Saturation augmentation
            hsv_v=0.4   # Value augmentation
        )
        
        training_time = time.time() - start_time
        
        # Save the final model
        model_save_path = 'models/bloomshield_yolov8m.pt'
        model.save(model_save_path)
        
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print(f"⏱️  Training time: {training_time/60:.2f} minutes")
        print(f"💾 Model saved to: {model_save_path}")
        
        # Display training results
        if results:
            print("\n📊 Training Results:")
            print(f"📈 Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"📈 Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"📉 Final loss: {results.results_dict.get('train/box_loss', 'N/A')}")
        
        # Training summary
        print("\n📝 Training Summary:")
        print(f"🏷️  Dataset: New Plant Diseases Dataset")
        print(f"🧠 Model: YOLOv8m")
        print(f"📊 Epochs completed: {EPOCHS}")
        print(f"💾 Model file: {model_save_path}")
        print(f"📁 Training logs: runs/train/bloomshield_yolov8m")
        
        return model_save_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("Please check your dataset and system requirements")
        return None

def validate_model(model_path):
    """Validate the trained model"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"\n🔍 Validating model: {model_path}")
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data='datasets/data.yaml')
        
        print("✓ Model validation completed")
        if results:
            print(f"📊 Validation mAP50: {results.box.map50:.4f}")
            print(f"📊 Validation mAP50-95: {results.box.map:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def create_training_log():
    """Create a training log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_content = f"""
BloomShield Training Log
========================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: YOLOv8m
Dataset: New Plant Diseases Dataset
Epochs: {EPOCHS}
Batch Size: {BATCH_SIZE}
Image Size: {IMAGE_SIZE}
Device: {DEVICE}

Training completed successfully!
Model saved as: models/bloomshield_yolov8m.pt

For deployment:
1. Use the saved model file in your Flask app
2. Update app.py to load this model
3. Test predictions with sample images

Next steps:
- Deploy on Render
- Add community upload features
- Create documentation
    """
    
    with open(f'training_log_{timestamp}.txt', 'w') as f:
        f.write(log_content)
    
    print(f"📄 Training log saved: training_log_{timestamp}.txt")

if __name__ == "__main__":
    print("🌱 BloomShield YOLOv8m Training")
    print("Building ML solution for crop disease detection")
    print("-" * 60)
    
    # Train the model
    model_path = train_bloomshield_model()
    
    if model_path and os.path.exists(model_path):
        # Validate the trained model
        validate_model(model_path)
        
        # Create training log
        create_training_log()
        
        print("\n🚀 Ready for deployment!")
        print("Next: Update app.py and create frontend")
    else:
        print("\n❌ Training failed. Please check your setup and try again.") 