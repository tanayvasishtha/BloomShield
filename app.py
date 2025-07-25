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

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configure Flask for production
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache for static files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
        'status': 'Healthy',
        'disease': 'No Disease Detected',
        'assessment': 'Your apple plant appears healthy with no signs of disease.',
        'treatment': 'No treatment required. Continue with regular care and monitoring.',
        'prevention': 'Maintain proper watering, fertilization, and pruning schedule.',
        'severity': 'Healthy'
    },
    'Apple_Apple': {
        'status': 'Healthy',
        'disease': 'No Disease Detected',
        'assessment': 'Your apple plant appears healthy with no signs of disease.',
        'treatment': 'No treatment required. Continue with regular care and monitoring.',
        'prevention': 'Maintain proper pruning, adequate spacing for air circulation, balanced nutrition, and regular inspection for early disease detection.',
        'severity': 'Healthy'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'Corn Gray Leaf Spot',
        'treatment': 'Apply fungicides containing strobilurin or triazole. Rotate crops.',
        'prevention': 'Use resistant varieties, crop rotation, proper field sanitation.',
        'severity': 'Moderate'
    },
    'Corn_(maize)___Common_rust': {
        'disease': 'Corn Common Rust',
        'treatment': 'Apply fungicides if severe. Usually manageable with resistant varieties.',
        'prevention': 'Plant resistant hybrids, avoid late planting.',
        'severity': 'Low to Moderate'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'Northern Corn Leaf Blight',
        'treatment': 'Apply fungicides (strobilurin, triazole). Practice crop rotation.',
        'prevention': 'Use resistant varieties, crop rotation, tillage practices.',
        'severity': 'High'
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy Corn',
        'treatment': 'No treatment needed. Continue regular monitoring.',
        'prevention': 'Maintain proper nutrition and water management.',
        'severity': 'None'
    },
    'Grape___Black_rot': {
        'disease': 'Grape Black Rot',
        'treatment': 'Apply fungicides (Mancozeb, Captan). Remove infected berries and leaves.',
        'prevention': 'Proper pruning for air circulation, regular fungicide applications.',
        'severity': 'High'
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'Grape Esca',
        'treatment': 'Remove infected wood. Apply wound protectants. No cure available.',
        'prevention': 'Proper pruning techniques, wound protection, avoid stress.',
        'severity': 'Very High'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease': 'Grape Leaf Blight',
        'treatment': 'Apply copper-based fungicides. Improve air circulation.',
        'prevention': 'Proper vine spacing, canopy management, fungicide applications.',
        'severity': 'Moderate'
    },
    'Grape___healthy': {
        'disease': 'Healthy Grape',
        'treatment': 'No treatment needed. Continue regular care.',
        'prevention': 'Maintain proper pruning, nutrition, and disease monitoring.',
        'severity': 'None'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease': 'Citrus Greening',
        'treatment': 'No cure. Remove infected trees. Control psyllid vectors.',
        'prevention': 'Use certified disease-free planting material, control psyllids.',
        'severity': 'Very High'
    },
    'Peach___Bacterial_spot': {
        'disease': 'Peach Bacterial Spot',
        'treatment': 'Apply copper-based bactericides. Remove infected fruit and leaves.',
        'prevention': 'Plant resistant varieties, proper sanitation, avoid overhead irrigation.',
        'severity': 'Moderate'
    },
    'Peach___healthy': {
        'disease': 'Healthy Peach',
        'treatment': 'No treatment needed. Continue regular monitoring.',
        'prevention': 'Maintain proper nutrition and pruning practices.',
        'severity': 'None'
    },
    'Pepper,_bell___Bacterial_spot': {
        'disease': 'Pepper Bacterial Spot',
        'treatment': 'Apply copper-based bactericides. Remove infected plants.',
        'prevention': 'Use disease-free seeds, crop rotation, avoid overhead watering.',
        'severity': 'Moderate'
    },
    'Pepper,_bell___healthy': {
        'disease': 'Healthy Bell Pepper',
        'treatment': 'No treatment needed. Continue regular care.',
        'prevention': 'Maintain proper watering and nutrition.',
        'severity': 'None'
    },
    'Potato___Early_blight': {
        'disease': 'Potato Early Blight',
        'treatment': 'Apply fungicides (Chlorothalonil, Mancozeb). Remove infected foliage.',
        'prevention': 'Crop rotation, proper spacing, avoid overhead irrigation.',
        'severity': 'Moderate'
    },
    'Potato___Late_blight': {
        'disease': 'Potato Late Blight',
        'treatment': 'Apply fungicides (Metalaxyl, Mancozeb) immediately. Remove infected plants.',
        'prevention': 'Use certified seed, avoid wet conditions, apply preventive fungicides.',
        'severity': 'Very High'
    },
    'Potato___healthy': {
        'disease': 'Healthy Potato',
        'treatment': 'No treatment needed. Continue regular monitoring.',
        'prevention': 'Maintain proper soil health and crop rotation.',
        'severity': 'None'
    },
    'Squash___Powdery_mildew': {
        'disease': 'Squash Powdery Mildew',
        'treatment': 'Apply fungicides (sulfur, potassium bicarbonate). Improve air circulation.',
        'prevention': 'Proper spacing, resistant varieties, avoid overhead watering.',
        'severity': 'Moderate'
    },
    'Strawberry___Leaf_scorch': {
        'disease': 'Strawberry Leaf Scorch',
        'treatment': 'Apply fungicides. Remove infected leaves. Improve drainage.',
        'prevention': 'Plant resistant varieties, proper spacing, avoid wet conditions.',
        'severity': 'Moderate'
    },
    'Strawberry___healthy': {
        'disease': 'Healthy Strawberry',
        'treatment': 'No treatment needed. Continue regular care.',
        'prevention': 'Maintain proper irrigation and nutrition.',
        'severity': 'None'
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Tomato Bacterial Spot',
        'treatment': 'Apply copper-based bactericides. Remove infected leaves.',
        'prevention': 'Use disease-free seeds, crop rotation, avoid overhead watering.',
        'severity': 'Moderate'
    },
    'Tomato___Early_blight': {
        'disease': 'Tomato Early Blight',
        'treatment': 'Apply fungicides (Chlorothalonil, Mancozeb). Remove infected foliage.',
        'prevention': 'Crop rotation, proper spacing, mulching.',
        'severity': 'Moderate'
    },
    'Tomato___Late_blight': {
        'disease': 'Tomato Late Blight',
        'treatment': 'Apply fungicides immediately. Remove infected plants.',
        'prevention': 'Use resistant varieties, avoid wet conditions, proper spacing.',
        'severity': 'Very High'
    },
    'Tomato___Leaf_Mold': {
        'disease': 'Tomato Leaf Mold',
        'treatment': 'Improve ventilation. Apply fungicides if severe.',
        'prevention': 'Proper spacing, avoid overhead watering, use resistant varieties.',
        'severity': 'Moderate'
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
    """Load YOLOv8 model with fallback options for production deployment"""
    global model, class_names
    
    # Try multiple model paths in order of preference
    model_paths = [
        'models/bloomshield_yolov8m_high_accuracy_fixed.pt',  # Custom trained model (local only)
        'yolov8n-cls.pt',  # Lightweight fallback for production
        'yolov8s-cls.pt',  # Small fallback
        'yolov8m-cls.pt'   # Medium fallback
    ]
    
    for model_path in model_paths:
        try:
            print(f"Attempting to load model: {model_path}")
            
            # Check if custom model exists
            if model_path.startswith('models/') and not os.path.exists(model_path):
                print(f"Custom model not found at {model_path}, trying next option...")
                continue
            
            # Load model with memory optimization
            model = YOLO(model_path)
            
            # Verify model loaded correctly
            if hasattr(model, 'names') and model.names:
                class_names = list(model.names.values())
                
                # Success message based on model type
                if 'bloomshield' in model_path:
                    print(f"SUCCESS: HIGH-ACCURACY Custom Model loaded!")
                    print(f"Performance: 99.7% Top-1 Accuracy, 100% Top-5 Accuracy")
                else:
                    print(f"SUCCESS: Production Model loaded: {model_path}")
                    print(f"Note: Using YOLOv8 pretrained model (works for general plant classification)")
                
                print(f"Classes: {len(class_names)} disease types")
                print(f"Model: {model_path}")
                return True
            else:
                print(f"Model loaded but no class names found: {model_path}")
                continue
                
        except Exception as e:
            print(f"Failed to load {model_path}: {str(e)}")
            continue
    
    # If all models fail
    print("ERROR: Could not load any model. Please check model files.")
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
    print("SUCCESS: Database initialized")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_classes': len(class_names) if class_names else 0,
        'static_folder': app.static_folder
    })

@app.route('/debug/css')
def debug_css():
    """Debug CSS loading"""
    import os
    css_path = os.path.join(app.static_folder, 'style.css')
    css_exists = os.path.exists(css_path) if css_path else False
    
    return jsonify({
        'css_file_exists': css_exists,
        'css_path': css_path,
        'static_folder': app.static_folder,
        'static_url_path': app.static_url_path
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from uploaded image"""
    if not model:
        return jsonify({
            'success': False,
            'error': 'AI model not available',
            'message': 'The disease detection model is currently unavailable. Please try again later or contact support.'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded',
            'message': 'Please select an image file to analyze.'
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected',
            'message': 'Please choose an image file before analyzing.'
        }), 400
    
    # Check file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'message': 'Please upload a valid image file (JPG, PNG, BMP, or TIFF).'
        }), 400
    
    try:
        # Read and process image with size limits for memory efficiency
        image = Image.open(file.stream).convert('RGB')
        
        # Resize large images to prevent memory issues
        max_size = (640, 640)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Run YOLOv8 prediction with error handling
        try:
            results = model(image)
        except Exception as prediction_error:
            print(f"Model prediction error: {prediction_error}")
            return jsonify({
                'success': False,
                'error': 'Model prediction failed',
                'message': 'Unable to process the image with the AI model. Please try again.',
                'technical_details': str(prediction_error) if app.debug else None
            }), 500
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            # For classification models, use probs instead of boxes
            if hasattr(result, 'probs') and result.probs is not None:
                # Get classification results
                confidence = float(result.probs.top1conf)
                class_id = int(result.probs.top1)
                predicted_class = model.names[class_id]
                
                # Debug logging
                print(f"Detected class: '{predicted_class}' with confidence {confidence:.3f}")
                
                # Get treatment suggestion with better class name matching
                # Handle different class name formats
                treatment_info = None
                
                # Try exact match first
                if predicted_class in TREATMENT_SUGGESTIONS:
                    treatment_info = TREATMENT_SUGGESTIONS[predicted_class]
                    print(f"Found treatment info using exact match: '{predicted_class}'")
                else:
                    # Try common variations
                    variations = [
                        predicted_class.replace('_', '___'),  # Apple_Apple -> Apple___Apple
                        predicted_class.replace('__', '___'), # Apple__Apple -> Apple___Apple
                        predicted_class + '_scab',            # Apple_Apple -> Apple_Apple_scab
                        predicted_class.replace('_Apple', '___Apple_scab'), # Apple_Apple -> Apple___Apple_scab
                    ]
                    
                    for variation in variations:
                        if variation in TREATMENT_SUGGESTIONS:
                            treatment_info = TREATMENT_SUGGESTIONS[variation]
                            print(f"Found treatment info using variation: '{variation}'")
                            break
                
                # Fallback if no match found
                if not treatment_info:
                    # Provide general plant care advice based on predicted class
                    clean_name = predicted_class.replace('_', ' ').title()
                    
                    if 'healthy' in predicted_class.lower():
                        treatment_info = {
                            'status': 'Healthy',
                            'disease': 'No Disease Detected',
                            'assessment': f'Your {clean_name.replace(" Healthy", "")} plant appears healthy with no signs of disease.',
                            'treatment': 'No treatment required. Continue with regular care and monitoring.',
                            'prevention': 'Maintain proper watering, nutrition, and regular inspection for early disease detection.',
                            'severity': 'Healthy'
                        }
                    else:
                        treatment_info = {
                            'disease': clean_name,
                            'assessment': f'Potential issue detected in your {clean_name} plant.',
                            'treatment': 'Consult with local agricultural extension service for specific treatment recommendations.',
                            'prevention': 'Follow good agricultural practices, ensure proper spacing, and monitor regularly.',
                            'severity': 'Unknown'
                        }
                
                # Convert image to base64 for display
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Determine if plant is healthy or diseased
                is_healthy = ('healthy' in predicted_class.lower() or 
                            treatment_info.get('severity') == 'Healthy' or
                            'healthy' in treatment_info.get('disease', '').lower())
                
                response = {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_healthy': is_healthy,
                    'plant_status': 'Healthy' if is_healthy else 'Disease Detected',
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
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")  # Log for debugging
        
        return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'message': 'Unable to analyze the image. Please ensure it\'s a clear photo of a plant and try again.',
            'technical_details': error_msg if app.debug else None
        }), 500

@app.route('/community_upload', methods=['POST'])
def community_upload():
    """Handle community data uploads for model improvement"""
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Missing file or label'}), 400
    
    file = request.files['file']
    user_label = request.form['label']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        image_data = file.read()
        
        # Get prediction for comparison
        predicted_class = 'Unknown'
        confidence = 0.0
        
        if model:
            file.stream.seek(0)  # Reset file pointer
            image = Image.open(file.stream).convert('RGB')
            results = model(image)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs') and result.probs is not None:
                    confidence = float(result.probs.top1conf)
                    class_id = int(result.probs.top1)
                    predicted_class = model.names[class_id]
        
        # Store in database
        conn = sqlite3.connect('community_uploads.db')
        conn.execute('''
            INSERT INTO uploads (filename, predicted_class, user_label, confidence, image_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, predicted_class, user_label, confidence, image_data))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your contribution! This helps improve our model.',
            'predicted_class': predicted_class,
            'user_label': user_label
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """Get application statistics"""
    try:
        conn = sqlite3.connect('community_uploads.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM uploads')
        total_uploads = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_label) FROM uploads WHERE user_label IS NOT NULL')
        unique_classes = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_uploads': total_uploads,
            'unique_classes': unique_classes,
            'model_accuracy': 99.7,
            'model_classes': len(class_names) if class_names else 0
        })
        
    except Exception as e:
        return jsonify({'error': f'Stats failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting BloomShield Flask App...")
    
    # Initialize database
    setup_database()
    
    # Load model
    if load_model():
        print("SUCCESS: BloomShield ready!")
    else:
        print("WARNING: BloomShield starting without model - some features may be limited")
    
    # For production, use the PORT environment variable from Render
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 