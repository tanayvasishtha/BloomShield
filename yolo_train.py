from ultralytics import YOLO
import os
import time
import shutil

def prepare_yolo_dataset():
    """Convert our data structure to YOLO classification format"""
    # Create YOLO dataset structure
    yolo_root = 'yolo_dataset'
    os.makedirs(f'{yolo_root}/train', exist_ok=True)
    os.makedirs(f'{yolo_root}/val', exist_ok=True)
    
    # Process training data
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract class name from filename (first part before underscore)
                class_name = img_file.split('_')[0]
                class_dir = f'{yolo_root}/train/{class_name}'
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(train_dir, img_file)
                dst = os.path.join(class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Process validation data
    val_dir = 'data/valid'
    if os.path.exists(val_dir):
        for img_file in os.listdir(val_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = img_file.split('_')[0]
                class_dir = f'{yolo_root}/val/{class_name}'
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_dir, img_file)
                dst = os.path.join(class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Create data.yaml
    classes = sorted(os.listdir(f'{yolo_root}/train'))
    yaml_content = f"""
path: {os.path.abspath(yolo_root)}
train: train
val: val
nc: {len(classes)}
names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}
"
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print("data.yaml created")
    return os.path.abspath('data.yaml')  # Return yaml path

def train_yolo_model():
    print("Starting YOLOv8 Classification Training...")
    
    # Prepare dataset
    dataset_yaml = prepare_yolo_dataset()
    
    # Load YOLOv8 classification model
    model = YOLO('yolov8n-cls.pt')  # Load YOLOv8 classification model
    
    print("Model loaded successfully!")
    print(f"Training on dataset: {dataset_yaml}")
    
    start_time = time.time()
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=10,
        imgsz=224,
        batch=16,
        device='cpu',
        project='runs/classify',
        name='bloomshield',
        verbose=True,
        patience=50,
        save=True,
        plots=True
    )
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save the trained model
    model.save('bloomshield_yolo_model.pt')
    print("Model saved as bloomshield_yolo_model.pt")
    
    return model, results

if __name__ == "__main__":
    model, results = train_yolo_model() 