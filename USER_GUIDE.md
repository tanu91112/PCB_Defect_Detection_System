# 👤 User Guide - PCB Defect Detection System

## 🎯 Getting Started

### Prerequisites
- Web browser (Chrome, Firefox, Safari, Edge)
- Template image (defect-free PCB)
- Test image (PCB with potential defects)

### Starting the Application
1. **Run the web server**
   ```bash
   python web_app.py
   ```
2. **Open your browser** and go to: `http://localhost:5000`

## 🖥️ Web Interface Guide

### Main Interface
The web interface consists of:
- **Upload Section**: Template and test image upload
- **Parameter Controls**: Detection sensitivity settings
- **Results Section**: Annotated images and predictions
- **Download Options**: Export results

### Step-by-Step Usage

#### 1. Upload Images
- **Template Image**: Upload a defect-free PCB image
- **Test Image**: Upload the PCB image to be analyzed
- Supported formats: JPG, PNG, JPEG
- Recommended size: 500x500 to 2000x2000 pixels

#### 2. Adjust Parameters
- **Difference Threshold (30)**: Sensitivity for detecting differences
  - Lower values = more sensitive (detects smaller changes)
  - Higher values = less sensitive (ignores minor variations)
- **Minimum Area (50)**: Minimum defect size in pixels
  - Smaller values = detect smaller defects
  - Larger values = ignore tiny noise
- **Confidence Threshold (0.6)**: Model confidence for classification
  - Higher values = more conservative predictions
  - Lower values = more predictions (may include false positives)

#### 3. Run Detection
- Click **"🔍 Detect Defects"** button
- Wait for processing (typically 2-5 seconds)
- View results in the results section

#### 4. Interpret Results
- **Defect Count**: Number of potential defects found
- **High Confidence**: Number of high-confidence predictions
- **Image Grid**: Shows template, test, difference, mask, and annotated images
- **Predictions Table**: Detailed defect information

#### 5. Download Results
- **Download Annotated Image**: Get the result image with bounding boxes
- **Download CSV Log**: Get detailed prediction data in spreadsheet format

## 📊 Understanding Results

### Image Types
1. **Template**: Your uploaded reference image
2. **Test**: Your uploaded test image
3. **Difference**: Highlighted differences between images
4. **Binary Mask**: Cleaned difference regions
5. **Annotated**: Final result with defect labels

### Defect Classes
- **Open**: Broken or missing connections
- **Short**: Unintended connections between traces
- **Mousebite**: Small circular defects
- **Spur**: Extra copper protrusions
- **Pinhole**: Small holes in copper
- **Spurious Copper**: Unwanted copper deposits

### Prediction Table Columns
- **Defect #**: Sequential defect number
- **Class**: Predicted defect type
- **Confidence**: Model confidence (0-1, higher is better)
- **Position**: Bounding box coordinates
- **Size**: Width and height of defect region

## ⚙️ Parameter Tuning Guide

### For Better Detection
- **Lower Difference Threshold** (20-25): More sensitive to small changes
- **Lower Minimum Area** (30-40): Detect smaller defects
- **Lower Confidence Threshold** (0.4-0.5): Include more predictions

### For Fewer False Positives
- **Higher Difference Threshold** (40-50): Less sensitive to noise
- **Higher Minimum Area** (80-100): Ignore small artifacts
- **Higher Confidence Threshold** (0.7-0.8): Only high-confidence predictions

### For Specific Defect Types
- **Open/Short**: Use lower thresholds (more sensitive)
- **Mousebite/Pinhole**: Use higher minimum area (ignore tiny noise)
- **Spur/Spurious Copper**: Use medium thresholds (balanced sensitivity)

## 🔧 Troubleshooting

### Common Issues

#### "No defects detected"
- **Solution**: Lower the difference threshold (try 20-25)
- **Check**: Ensure images are properly aligned
- **Try**: Lower minimum area (30-40)

#### "Too many false positives"
- **Solution**: Increase difference threshold (40-50)
- **Try**: Increase minimum area (80-100)
- **Adjust**: Raise confidence threshold (0.7-0.8)

#### "Model not loaded" error
- **Check**: Ensure `training_outputs/model_best.pth` exists
- **Solution**: Run training first or check file paths

#### "Invalid image format" error
- **Check**: Image file is not corrupted
- **Try**: Convert to JPG format
- **Ensure**: File size is reasonable (< 10MB)

#### Slow processing
- **Check**: Image size (resize if > 2000x2000)
- **Try**: Close other applications
- **Consider**: Using GPU if available

### Performance Tips
1. **Image Quality**: Use high-quality, well-lit images
2. **Alignment**: Ensure template and test images are properly aligned
3. **Size**: Resize large images to 1000x1000 for faster processing
4. **Format**: Use JPG for smaller file sizes

## 📱 Mobile Usage

The web interface is responsive and works on mobile devices:
- Upload images from your phone's camera
- Adjust parameters using touch controls
- Download results directly to your device

## 💡 Best Practices

### Image Preparation
1. **Good Lighting**: Ensure even, bright lighting
2. **Stable Camera**: Use tripod or stable surface
3. **Clean Images**: Remove dust and reflections
4. **Consistent Angle**: Keep same camera angle for template and test

### Parameter Selection
1. **Start Default**: Begin with default parameters
2. **Adjust Gradually**: Make small changes (5-10 units)
3. **Test Multiple**: Try different parameter combinations
4. **Document Results**: Keep notes on what works best

### Workflow Optimization
1. **Batch Processing**: Process multiple images with same parameters
2. **Save Settings**: Note parameter values that work well
3. **Regular Testing**: Test with known good/bad samples
4. **Update Model**: Retrain with new data if needed

## 📞 Support

If you encounter issues:
1. Check this user guide first
2. Try the troubleshooting section
3. Check the technical documentation
4. Create an issue in the repository

## 🔄 Updates

The system is regularly updated with:
- Improved accuracy
- New features
- Bug fixes
- Performance optimizations

Check the repository for the latest version and updates.
