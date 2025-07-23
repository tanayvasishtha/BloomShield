import sys
sys.path.append('yolov9')
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import cv2
import numpy as np
import sqlite3
import os

from train import SimpleCNN, class_to_idx  # Import model and class mapping

app = Flask(__name__)

device = select_device('cpu')
model = DetectMultiBackend('bloomshield_yolo_model.pt', device=device)
model = model.model
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple treatment suggestions (rule-based)
treatment_suggestions = {
    # Add suggestions for each class, e.g.
    'Apple_healthy': 'Your apple plant is healthy. Maintain good care.',
    'Corn_(maize)_Northern_Leaf_Blight': 'Apply fungicide and remove affected leaves.',
    # ... Add for all 38 classes as needed
}

# SQLite setup for community uploads
conn = sqlite3.connect('uploads.db')
conn.execute('CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY, image BLOB, label TEXT)')
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)
    output = model(img)
    pred = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)[0]
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img.shape[2:]).round()
        class_id = int(pred[0, 5])
        class_name = model.names[class_id]
        bboxes = pred[:, :4].cpu().numpy().tolist()
    else:
        class_name = 'No detection'
        bboxes = []
    suggestion = treatment_suggestions.get(class_name, 'No suggestion available.')
    return jsonify({'class': class_name, 'suggestion': suggestion, 'bboxes': bboxes})

@app.route('/community_upload', methods=['POST'])
def community_upload():
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Missing file or label'})
    file = request.files['file']
    label = request.form['label']
    conn = sqlite3.connect('uploads.db')
    conn.execute('INSERT INTO uploads (image, label) VALUES (?, ?)', (file.read(), label))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Upload successful'})

if __name__ == '__main__':
    app.run(debug=True) 