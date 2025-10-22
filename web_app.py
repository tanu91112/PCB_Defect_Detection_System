#!/usr/bin/env python3
"""
GOAL: Web interface for PCB defect detection and visualization
- Upload template and test images for defect detection
- Run preprocessing pipeline to extract defect ROIs
- Classify defects using trained EfficientNet-B4 model
- Display results with bounding boxes and class labels
- Download annotated images and detection reports
"""

from flask import Flask, render_template, request, jsonify, send_file, make_response
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import json
from pathlib import Path
import base64
from PIL import Image
import io
import os
import csv
import time
from datetime import datetime

app = Flask(__name__)

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFECT_CLASSES = {
    1: "open", 2: "short", 3: "mousebite", 
    4: "spur", 5: "pinhole", 6: "spurious copper"
}

# Global model cache
model = None
classes = None

def load_model():
    """Load the trained EfficientNet model"""
    global model, classes
    try:
        model_path = "training_outputs/model_best.pth"
        classes_path = "training_outputs/classes.json"
        
        if not Path(model_path).exists() or not Path(classes_path).exists():
            return False
            
        # Load classes
        with open(classes_path, 'r') as f:
            classes = json.load(f)
            
        # Build model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, len(classes)) # type: ignore
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image, img_size=128):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image).unsqueeze(0)

def to_gray(img):
    """Convert image to grayscale"""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_gray(gray):
    """Preprocess grayscale image"""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    g = cv2.equalizeHist(g)
    return g

def absdiff_norm(template_gray, test_gray):
    """Compute absolute difference between template and test"""
    return cv2.absdiff(test_gray, template_gray)

def mask_from_diff(diff, thresh=30):
    """Create binary mask from difference image"""
    _, th = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated

def extract_rois(mask, min_area=50):
    """Extract ROI bounding boxes from mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area:
            rois.append((x, y, w, h))
    rois.sort(key=lambda b: b[2] * b[3], reverse=True)
    return rois

def predict_defect(model, image_crop, classes, device):
    """Predict defect class for a single ROI"""
    with torch.no_grad():
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        input_tensor = preprocess_image(pil_image).to(device)
        
        # Predict
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        
        pred_class = classes[pred_idx.item()]
        confidence_score = confidence.item()
        
        return pred_class, confidence_score

def annotate_image(image, rois, predictions):
    """Annotate image with predictions"""
    annotated = image.copy()
    
    # Draw predictions in red
    for (x, y, w, h), (pred_class, confidence) in zip(rois, predictions):
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(annotated, f"{pred_class} ({confidence:.2f})", 
                   (x, min(annotated.shape[0]-2, y+h+16)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return annotated

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_defects():
    """API endpoint for defect detection"""
    try:
        start_time = time.time()
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get parameters
        thresh = int(request.form.get('thresh', 30))
        min_area = int(request.form.get('min_area', 50))
        conf_thresh = float(request.form.get('conf_thresh', 0.6))
        
        # Get uploaded files
        template_file = request.files['template']
        test_file = request.files['test']
        
        if not template_file or not test_file:
            return jsonify({'error': 'Both template and test images required'}), 400
        
        # Read images
        template_bytes = template_file.read()
        test_bytes = test_file.read()
        
        template_np = np.frombuffer(template_bytes, np.uint8)
        test_np = np.frombuffer(test_bytes, np.uint8)
        
        template_img = cv2.imdecode(template_np, cv2.IMREAD_COLOR)
        test_img = cv2.imdecode(test_np, cv2.IMREAD_COLOR)
        
        if template_img is None or test_img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Preprocessing
        template_gray = preprocess_gray(to_gray(template_img))
        test_gray = preprocess_gray(to_gray(test_img))
        
        # Difference and mask
        diff = absdiff_norm(template_gray, test_gray)
        mask = mask_from_diff(diff, thresh)
        
        # Extract ROIs
        rois = extract_rois(mask, min_area)
        
        if len(rois) == 0:
            return jsonify({
                'success': True,
                'defects_found': 0,
                'message': 'No defects detected. Try adjusting parameters.',
                'images': {
                    'template': image_to_base64(template_img),
                    'test': image_to_base64(test_img),
                    'diff': image_to_base64(diff),
                    'mask': image_to_base64(mask)
                }
            })
        
        # Predict defects
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = []
        high_conf_predictions = []
        
        for x, y, w, h in rois:
            crop = test_img[y:y+h, x:x+w]
            if crop.size == 0:
                continue
                
            pred_class, confidence = predict_defect(model, crop, classes, device)
            predictions.append((pred_class, confidence))
            
            if confidence >= conf_thresh:
                high_conf_predictions.append({
                    'bbox': [x, y, w, h],
                    'class': pred_class,
                    'confidence': confidence
                })
        
        # Annotate image
        annotated_img = annotate_image(test_img, rois, predictions)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Store results for download
        session_id = str(int(time.time()))
        results = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'defects': high_conf_predictions,
            'processing_time': processing_time,
            'annotated_image': image_to_base64(annotated_img)
        }
        
        # Save results to temporary file
        results_file = f"temp_results_{session_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        return jsonify({
            'success': True,
            'defects_found': len(rois),
            'high_confidence': len(high_conf_predictions),
            'predictions': high_conf_predictions,
            'session_id': session_id,
            'images': {
                'template': image_to_base64(template_img),
                'test': image_to_base64(test_img),
                'diff': image_to_base64(diff),
                'mask': image_to_base64(mask),
                'annotated': image_to_base64(annotated_img)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_image/<session_id>')
def download_image(session_id):
    """Download annotated image"""
    try:
        results_file = f"temp_results_{session_id}.json"
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Decode base64 image
        img_data = base64.b64decode(results['annotated_image'].split(',')[1])
        
        # Create response
        response = make_response(img_data)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Content-Disposition'] = f'attachment; filename=annotated_result_{session_id}.jpg'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_log/<session_id>')
def download_log(session_id):
    """Download CSV log of predictions"""
    try:
        results_file = f"temp_results_{session_id}.json"
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Session ID', 'Defect ID', 'Class', 'Confidence', 'Bounding Box (x1,y1,x2,y2)'])
        
        # Write defect data
        for i, defect in enumerate(results['defects']):
            writer.writerow([
                results['timestamp'],
                session_id,
                i + 1,
                defect['class'],
                f"{defect['confidence']:.4f}",
                f"({defect['bbox'][0]},{defect['bbox'][1]},{defect['bbox'][2]},{defect['bbox'][3]})"
            ])
        
        # Create response
        csv_data = output.getvalue()
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=prediction_log_{session_id}.csv'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup/<session_id>')
def cleanup(session_id):
    """Clean up temporary files"""
    try:
        results_file = f"temp_results_{session_id}.json"
        if os.path.exists(results_file):
            os.remove(results_file)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Load model
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"Classes: {', '.join(classes)}") # type: ignore
    else:
        print("❌ Failed to load model")
        exit(1)
    
    print("🚀 Starting PCB Defect Detection Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
