from ultralytics import YOLO
import os
import time
import shutil
from PIL import Image

def prepare_yolo_dataset():
    """Convert our data structure to YOLO classification format"""
    print("Preparing dataset...")
    # Create YOLO dataset structure
    yolo_root = 'yolo_dataset'
    os.makedirs(f'{yolo_root}/train', exist_ok=True)
    os.makedirs(f'{yolo_root}/val', exist_ok=True)
    
    # Process training data
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            src = os.path.join(train_dir, img_file)
            try:
                img = Image.open(src)
                img.verify()  # Verify it's a valid image
            except:
                print(f"Skipping invalid image: {src}")
                continue
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract class name from filename (first part before underscore)
                class_name = img_file.split('_')[0]
                class_dir = f'{yolo_root}/train/{class_name}'
                os.makedirs(class_dir, exist_ok=True)
                
                dst = os.path.join(class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Process validation data
    val_dir = 'data/valid'
    if os.path.exists(val_dir):
        for img_file in os.listdir(val_dir):
            src = os.path.join(val_dir, img_file)
            try:
                img = Image.open(src)
                img.verify()
            except:
                print(f"Skipping invalid image: {src}")
                continue
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = img_file.split('_')[0]
                class_dir = f'{yolo_root}/val/{class_name}'
                os.makedirs(class_dir, exist_ok=True)
                
                dst = os.path.join(class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Create data.yaml
    classes = sorted(os.listdir(f'{yolo_root}/train'))
    print(f"Number of classes: {len(classes)}")
    print(f"Processed {len(os.listdir(f'{yolo_root}/train'))} train classes")
    print(f"Processed {len(os.listdir(f'{yolo_root}/val'))} val classes")
    for cls in os.listdir(f'{yolo_root}/train'):
        print(f"Class {cls}: {len(os.listdir(f'{yolo_root}/train/{cls}'))} images")
    for cls in os.listdir(f'{yolo_root}/val'):
        print(f"Class {cls}: {len(os.listdir(f'{yolo_root}/val/{cls}'))} images")
    return os.path.abspath(yolo_root)  # Return folder path

def train_yolo_model():
    try:
        print("Starting YOLOv8 Classification Training...")
    
        # Prepare dataset
        dataset_path = prepare_yolo_dataset()
    
        # Load YOLOv8 classification model
        model = YOLO('yolov8n-cls.pt')  # Load YOLOv8 classification model
    
        print("Model loaded successfully!")
        print(f"Training on dataset: {dataset_path}")
    
        start_time = time.time()
    
        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=1,  # Test with 1 epoch
            imgsz=224,
            batch=4,  # Smaller batch for testing
            device='cpu',
            project='runs/classify',
            name='bloomshield_test',
            verbose=True,
            patience=0,
            save=True,
            plots=True,
            augment=False  # Disable augmentation to avoid error
        )
    
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
    
        # Save the trained model
        model.save('bloomshield_yolo_model.pt')
        print("Model saved as bloomshield_yolo_model.pt")
    
        return model, results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    model, results = train_yolo_model() 