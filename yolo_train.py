import sys
sys.path.append('yolov9')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import torch
import time
import os
import shutil
from PIL import Image
import numpy as np
import torch.nn as nn
from pathlib import Path

class_names = None  # Will set later

def generate_labels(yolo_root):
    global class_names
    class_names = sorted(os.listdir(f'{yolo_root}/train'))
    for split in ['train', 'val']:
        for cls in class_names:
            img_dir = f'{yolo_root}/{split}/{cls}'
            label_dir = f'{yolo_root}/labels/{split}/{cls}'
            os.makedirs(label_dir, exist_ok=True)
            for img_file in os.listdir(img_dir):
                label_file = Path(img_file).stem + '.txt'
                label_path = os.path.join(label_dir, label_file)
                class_id = class_names.index(cls)
                # Dummy bbox: full image (center x=0.5, y=0.5, width=1, height=1)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1 1\n")
    print("Labels generated with dummy bounding boxes")

def prepare_yolo_dataset():
    print("Preparing dataset...")
    yolo_root = 'yolo_dataset'
    os.makedirs(f'{yolo_root}/train', exist_ok=True)
    os.makedirs(f'{yolo_root}/val', exist_ok=True)
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            src = os.path.join(train_dir, img_file)
            try:
                img = Image.open(src)
                img.verify()
            except:
                print(f"Skipping invalid image: {src}")
                continue
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = img_file.split('_')[0]
                class_dir = f'{yolo_root}/train/{class_name}'
                os.makedirs(class_dir, exist_ok=True)
                dst = os.path.join(class_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
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
    generate_labels(yolo_root)
    classes = len(class_names)
    yaml_content = f'''path: {Path(yolo_root).absolute()}\ntrain: ../{yolo_root}/train\nval: ../{yolo_root}/val\nnames:\n'''
    for i, cls in enumerate(class_names):
        yaml_content += f'  {i}: {cls}\n'
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print("data.yaml created")
    return 'data.yaml'

def train_yolo_model():
    dataset_yaml = prepare_yolo_dataset()
    model = YOLO("yolov9m.pt")
    results = model.train(data=dataset_yaml, epochs=10, imgsz=640, batch=8, device='cpu', name='bloomshield', verbose=True)
    model.save('bloomshield_yolo_model.pt')
    print("Model saved")

if __name__ == "__main__":
    train_yolo_model() 