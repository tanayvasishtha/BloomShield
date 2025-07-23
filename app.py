from flask import Flask, request, jsonify, render_template
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import sqlite3
import os

from train import SimpleCNN, class_to_idx  # Import model and class mapping

app = Flask(__name__)

# Load model
num_classes = len(class_to_idx)
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('bloomshield_model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()
    class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(pred)]
    suggestion = treatment_suggestions.get(class_name, 'No suggestion available.')
    return jsonify({'class': class_name, 'suggestion': suggestion})

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