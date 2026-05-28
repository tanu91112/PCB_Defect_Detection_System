# 🔧 Technical Documentation - PCB Defect Detection System

## 🏗️ System Architecture

### Overview
The system consists of three main components:
1. **Preprocessing Pipeline**: Image processing and ROI extraction
2. **Deep Learning Model**: EfficientNet-B4 classifier for defect classification
3. **Web Application**: FastAPI-based interface for real-time inference

### Technology Stack
- **Backend**: Python 3.11, FastAPI, OpenCV, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, NumPy
- **Visualization**: Matplotlib, scikit-learn

## 📁 Code Structure

### Core Modules

#### `src/preprocessing.py`
**Purpose**: Image preprocessing and ROI extraction pipeline

**Key Functions**:
- `load_image()`: Load images with error handling
- `to_gray()`: Convert to grayscale
- `preprocess_gray()`: Apply blur and histogram equalization
- `subtract_images()`: Calculate absolute difference
- `get_binary_mask()`: Create binary mask using thresholding and morphology
- `extract_rois()`: Find contours and extract bounding boxes
- `annotate_image()`: Draw bounding boxes and class labels
- `run_pipeline()`: Complete preprocessing workflow

**Parameters**:
- `thresh`: Difference threshold (default: 30)
- `min_area`: Minimum ROI area in pixels (default: 50)

#### `src/build_dataset.py`
**Purpose**: Create labeled dataset from DeepPCB annotations

**Key Functions**:
- `read_labels()`: Parse annotation files
- `collect_groups()`: Find dataset groups
- `crop_and_save_rois()`: Extract defect regions
- `stratified_split()`: Split dataset maintaining class distribution
- `build_dataset()`: Complete dataset creation workflow

**Output**: `dataset/{train,val,test}/{class_name}/` structure

#### `src/train_efficientnet_b4.py`
**Purpose**: Train EfficientNet-B4 classifier

**Key Functions**:
- `create_dataloaders()`: Create train/val/test dataloaders with augmentation
- `build_model()`: Create EfficientNet-B4 with custom classifier
- `train_model()`: Training loop with validation and early stopping
- `plot_curves()`: Generate training/validation curves
- `evaluate_confusion()`: Generate confusion matrix

**Training Parameters**:
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Epochs: 20
- Batch size: 32
- Image size: 128x128

#### `src/evaluate_model.py`
**Purpose**: Evaluate trained model on test set

**Key Functions**:
- `create_test_dataloader()`: Create test dataloader
- `build_model()`: Reconstruct model architecture
- `evaluate_model()`: Test model and generate metrics

**Outputs**:
- Test accuracy and per-class metrics
- Confusion matrix visualization
- Detailed results JSON

#### `web_app.py`
**Purpose**: FastAPI web application for real-time inference

**Routes**:
- `GET /`: Main interface
- `POST /detect`: Defect detection API
- `GET /download_image/<session_id>`: Download annotated image
- `GET /download_log/<session_id>`: Download CSV log
- `GET /cleanup/<session_id>`: Clean up temporary files

## 🧠 Model Architecture

### EfficientNet-B4
- **Base Model**: EfficientNet-B4 (ImageNet pretrained)
- **Input Size**: 128x128x3
- **Output**: 6 classes (defect types)
- **Parameters**: ~19M trainable parameters

### Data Augmentation
```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Training Configuration
- **Optimizer**: Adam (lr=1e-4, weight_decay=0)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: None (constant learning rate)
- **Early Stopping**: Based on validation accuracy
- **Batch Size**: 32
- **Epochs**: 20

## 🔄 Data Flow

### Training Pipeline
1. **Dataset Creation**: `build_dataset.py`
   - Parse DeepPCB annotations
   - Crop defect regions
   - Split into train/val/test (70/15/15)
2. **Model Training**: `train_efficientnet_b4.py`
   - Load pretrained EfficientNet-B4
   - Fine-tune on defect dataset
   - Save best model checkpoint
3. **Evaluation**: `evaluate_model.py`
   - Test on unseen data
   - Generate performance metrics

### Inference Pipeline
1. **Image Upload**: User uploads template and test images
2. **Preprocessing**: Extract differences and ROIs
3. **Classification**: Run EfficientNet-B4 on each ROI
4. **Post-processing**: Filter by confidence threshold
5. **Visualization**: Draw bounding boxes and labels
6. **Export**: Generate downloadable results

## 📊 Performance Metrics

### Model Performance
- **Overall Test Accuracy**: 98.34%
- **Per-class Accuracy**:
  - mousebite: 97.64%
  - open: 96.92%
  - pinhole: 100.00%
  - short: 99.56%
  - spur: 97.96%
  - spurious copper: 98.67%

### Processing Speed
- **Preprocessing**: ~0.5-1.0 seconds
- **Inference**: ~0.1-0.3 seconds per ROI
- **Total Time**: ~2-5 seconds per image pair

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **GPU**: CUDA-compatible (optional, for faster training)
- **Storage**: 2GB+ for model and data

## 🔧 API Reference

### Web API Endpoints

#### `POST /detect`
**Purpose**: Detect defects in uploaded images

**Request**:
- `template`: Template image file
- `test`: Test image file
- `thresh`: Difference threshold (default: 30)
- `min_area`: Minimum ROI area (default: 50)
- `conf_thresh`: Confidence threshold (default: 0.6)

**Response**:
```json
{
  "success": true,
  "defects_found": 5,
  "high_confidence": 3,
  "session_id": "1234567890",
  "predictions": [...],
  "images": {
    "template": "data:image/jpeg;base64,...",
    "test": "data:image/jpeg;base64,...",
    "diff": "data:image/jpeg;base64,...",
    "mask": "data:image/jpeg;base64,...",
    "annotated": "data:image/jpeg;base64,..."
  }
}
```

#### `GET /download_image/<session_id>`
**Purpose**: Download annotated result image

**Response**: JPEG image file

#### `GET /download_log/<session_id>`
**Purpose**: Download CSV prediction log

**Response**: CSV file with columns:
- Timestamp, Session ID, Defect ID
- Class, Confidence, Bounding Box

### Command Line Interface

#### Preprocessing
```bash
python src/preprocessing.py -t template.jpg -s test.jpg -o output_dir
```

#### Dataset Creation
```bash
python src/build_dataset.py --data-root data --out-root dataset
```

#### Training
```bash
python src/train_efficientnet_b4.py --data dataset --epochs 20 --batch-size 32
```

#### Evaluation
```bash
python src/evaluate_model.py --data dataset --model training_outputs/model_best.pth
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Check if model file exists
import os
print(os.path.exists("training_outputs/model_best.pth"))

# Check if classes file exists
print(os.path.exists("training_outputs/classes.json"))
```

#### CUDA Issues
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

#### Memory Issues
- Reduce batch size: `--batch-size 16`
- Reduce image size: `--img-size 96`
- Use CPU only: `--device cpu`

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

#### GPU Acceleration
```python
# Check GPU usage
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

#### Memory Management
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

#### Batch Processing
```python
# Process multiple images
for image_path in image_list:
    result = process_image(image_path)
    save_results(result)
```

## 🔄 Development Workflow

### Adding New Features
1. **Create feature branch**
2. **Implement changes**
3. **Test thoroughly**
4. **Update documentation**
5. **Create pull request**

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python test_integration.py

# Test web app
python web_app.py
# Open browser to http://localhost:5000
```

### Deployment
1. **Prepare environment**
2. **Install dependencies**
3. **Train model**
4. **Start web server**
5. **Configure reverse proxy** (optional)

## 📈 Monitoring and Logging

### Performance Monitoring
- Processing time tracking
- Memory usage monitoring
- GPU utilization (if available)
- Error rate tracking

### Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🔒 Security Considerations

### Input Validation
- File type validation
- File size limits
- Image format verification
- Path traversal protection

### Data Privacy
- Temporary file cleanup
- No persistent storage of uploaded images
- Session-based result management

## 📚 References

### Papers
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- DeepPCB: A Deep Learning Approach for PCB Defect Detection

### Libraries
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/
- FastAPI: https://fastAPI.palletsprojects.com/

### Datasets
- DeepPCB Dataset: PCB defect detection benchmark
