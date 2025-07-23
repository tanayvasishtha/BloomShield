"""
BloomShield Flask Backend
YOLOv8m-powered crop disease detection web application
"""

from flask import Flask, request, jsonify, render_template
import torch
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import sqlite3
import os
import base64
from datetime import datetime

app = Flask(__name__)

# Global model variable
model = None
class_names = []

# Disease treatment suggestions database
TREATMENT_SUGGESTIONS = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'treatment': 'Apply fungicide (Captan or Mancozeb) during early spring. Remove fallen leaves and improve air circulation.',
        'prevention': 'Plant resistant varieties, ensure proper spacing, prune for air circulation.',
        'severity': 'Moderate'
    },
    'Apple___Black_rot': {
        'disease': 'Apple Black Rot',
        'treatment': 'Remove infected fruit and cankers. Apply copper-based fungicides during dormant season.',
        'prevention': 'Proper pruning, avoid wounding trees, maintain tree health.',
        'severity': 'High'
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'Cedar Apple Rust',
        'treatment': 'Apply preventive fungicides (Myclobutanil). Remove nearby cedar trees if possible.',
        'prevention': 'Plant resistant apple varieties, maintain distance from cedar trees.',
        'severity': 'Moderate'
    },
    'Apple___healthy': {
        'disease': 'Healthy Apple',
        'treatment': 'No treatment needed. Continue regular care and monitoring.',
        'prevention': 'Maintain proper watering, fertilization, and pruning schedule.',
        'severity': 'None'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'Corn Gray Leaf Spot',
        'treatment': 'Apply foliar fungicides (Strobilurin-based). Improve field drainage.',
        'prevention': 'Crop rotation, resistant varieties, proper plant spacing.',
        'severity': 'Moderate'
    },
    'Corn_(maize)___Common_rust_': {
        'disease': 'Corn Common Rust',
        'treatment': 'Apply fungicides if severe. Usually not economically damaging.',
        'prevention': 'Plant resistant hybrids, avoid late planting.',
        'severity': 'Low'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'Northern Leaf Blight',
        'treatment': 'Apply fungicides (Azoxystrobin, Propiconazole). Remove crop residue.',
        'prevention': 'Crop rotation, tillage, resistant varieties.',
        'severity': 'High'
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy Corn',
        'treatment': 'No treatment needed. Monitor for early signs of disease.',
        'prevention': 'Regular field scouting, proper nutrition, water management.',
        'severity': 'None'
    },
    'Grape___Black_rot': {
        'disease': 'Grape Black Rot', 
        'treatment': 'Apply fungicides (Mancozeb, Captan) from bloom to harvest.',
        'prevention': 'Prune for air circulation, remove mummified berries.',
        'severity': 'High'
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'Grape Esca',
        'treatment': 'No cure available. Remove infected wood, protect pruning cuts.',
        'prevention': 'Avoid large pruning cuts, use trunk renewal techniques.',
        'severity': 'Very High'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease': 'Grape Leaf Blight',
        'treatment': 'Apply copper-based fungicides. Improve air circulation.',
        'prevention': 'Proper canopy management, avoid overhead irrigation.',
        'severity': 'Moderate'
    },
    'Grape___healthy': {
        'disease': 'Healthy Grape',
        'treatment': 'No treatment needed. Continue preventive care.',
        'prevention': 'Regular pruning, proper nutrition, disease monitoring.',
        'severity': 'None'
    },
    'Potato___Early_blight': {
        'disease': 'Potato Early Blight',
        'treatment': 'Apply fungicides (Chlorothalonil, Mancozeb). Remove infected foliage.',
        'prevention': 'Crop rotation, avoid overhead watering, proper spacing.',
        'severity': 'Moderate'
    },
    'Potato___Late_blight': {
        'disease': 'Potato Late Blight',
        'treatment': 'Apply fungicides immediately (Metalaxyl, Cymoxanil). Destroy infected plants.',
        'prevention': 'Use certified seed, avoid wet conditions, early detection.',
        'severity': 'Very High'
    },
    'Potato___healthy': {
        'disease': 'Healthy Potato',
        'treatment': 'No treatment needed. Monitor for disease symptoms.',
        'prevention': 'Proper crop rotation, avoid water stress, regular scouting.',
        'severity': 'None'
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Tomato Bacterial Spot',
        'treatment': 'Apply copper-based bactericides. Remove infected plants.',
        'prevention': 'Use pathogen-free seeds, avoid overhead irrigation.',
        'severity': 'High'
    },
    'Tomato___Early_blight': {
        'disease': 'Tomato Early Blight',
        'treatment': 'Apply fungicides (Chlorothalonil). Remove lower infected leaves.',
        'prevention': 'Proper spacing, avoid overhead watering, crop rotation.',
        'severity': 'Moderate'
    },
    'Tomato___Late_blight': {
        'disease': 'Tomato Late Blight',
        'treatment': 'Apply fungicides immediately (Metalaxyl). Remove infected plants.',
        'prevention': 'Avoid wet conditions, proper air circulation, resistant varieties.',
        'severity': 'Very High'
    },
    'Tomato___Leaf_Mold': {
        'disease': 'Tomato Leaf Mold',
        'treatment': 'Improve ventilation, apply fungicides if severe.',
        'prevention': 'Proper greenhouse ventilation, avoid high humidity.',
        'severity': 'Low'
    },
    'Tomato___Septoria_leaf_spot': {
        'disease': 'Tomato Septoria Leaf Spot',
        'treatment': 'Apply fungicides (Chlorothalonil, Mancozeb). Remove infected foliage.',
        'prevention': 'Avoid overhead watering, mulch around plants.',
        'severity': 'Moderate'
    },
    'Tomato___healthy': {
        'disease': 'Healthy Tomato',
        'treatment': 'No treatment needed. Continue regular care.',
        'prevention': 'Proper watering, nutrition, and disease monitoring.',
        'severity': 'None'
    }
}

def load_model():
    """Load YOLOv8m model"""
    global model, class_names
    
    model_path = 'models/bloomshield_yolov8m.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train.py")
        return False
    
    try:
        model = YOLO(model_path)
        class_names = list(model.names.values())
        print(f"✅ Model loaded successfully with {len(class_names)} classes")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def setup_database():
    """Initialize SQLite database for community uploads"""
    conn = sqlite3.connect('community_uploads.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_class TEXT,
            user_label TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_data BLOB
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from uploaded image"""
    if not model:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Run YOLOv8m prediction
        results = model(image)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Get highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                best_idx = np.argmax(confidences)
                best_class_id = int(classes[best_idx])
                confidence = float(confidences[best_idx])
                
                predicted_class = model.names[best_class_id]
                
                # Get treatment suggestion
                treatment_info = TREATMENT_SUGGESTIONS.get(predicted_class, {
                    'disease': predicted_class.replace('_', ' ').title(),
                    'treatment': 'Consult with local agricultural extension service for specific treatment recommendations.',
                    'prevention': 'Follow good agricultural practices and regular monitoring.',
                    'severity': 'Unknown'
                })
                
                # Convert image to base64 for display
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                response = {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': round(confidence * 100, 2),
                    'disease_info': treatment_info,
                    'image_data': f"data:image/jpeg;base64,{img_base64}",
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return jsonify(response)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No disease detected in the image',
                    'message': 'Please upload a clear image of plant leaves or symptoms'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction failed',
                'message': 'Unable to process the image'
            })
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/community_upload', methods=['POST'])
def community_upload():
    """Handle community data uploads for model improvement"""
    if 'file' not in request.files or 'user_label' not in request.form:
        return jsonify({'error': 'Missing file or label'}), 400
    
    file = request.files['file']
    user_label = request.form['user_label']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Get prediction for comparison
        image = Image.open(file.stream).convert('RGB')
        predicted_class = ''
        confidence = 0.0
        
        if model:
            results = model(image)
            if results and len(results) > 0 and results[0].boxes is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                if len(confidences) > 0:
                    best_idx = np.argmax(confidences)
                    predicted_class = model.names[int(classes[best_idx])]
                    confidence = float(confidences[best_idx])
        
        # Save to database
        file.stream.seek(0)  # Reset file pointer
        image_data = file.read()
        
        conn = sqlite3.connect('community_uploads.db')
        conn.execute('''
            INSERT INTO uploads (filename, predicted_class, user_label, confidence, image_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, predicted_class, user_label, confidence, image_data))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Thank you for contributing to BloomShield!',
            'predicted_class': predicted_class,
            'user_label': user_label,
            'confidence': round(confidence * 100, 2) if confidence > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Get application statistics"""
    try:
        conn = sqlite3.connect('community_uploads.db')
        cursor = conn.cursor()
        
        # Get upload count
        cursor.execute('SELECT COUNT(*) FROM uploads')
        upload_count = cursor.fetchone()[0]
        
        # Get unique classes
        cursor.execute('SELECT COUNT(DISTINCT user_label) FROM uploads WHERE user_label IS NOT NULL')
        unique_classes = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_predictions': 'N/A',  # Would need to track this separately
            'community_uploads': upload_count,
            'classes_detected': len(class_names) if class_names else 0,
            'unique_user_labels': unique_classes,
            'model_version': 'YOLOv8m'
        })
        
    except Exception as e:
        return jsonify({'error': f'Stats unavailable: {str(e)}'}), 500

if __name__ == '__main__':
    print("🌱 BloomShield Flask Server Starting...")
    print("=" * 50)
    
    # Setup database
    setup_database()
    
    # Load model
    if load_model():
        print("🚀 Server ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Server failed to start. Please train the model first.")
        print("Run: python train.py") 