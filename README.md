# 🔍 CircuitGuard: PCB Defect Detection System

CircuitGuard is an AI-powered PCB defect detection system that automates the identification of manufacturing defects using a hybrid pipeline of advanced computer vision and EfficientNet-B4 deep learning. The system provides real-time defect analysis through a FastAPI-based web interface, offering annotated outputs and downloadable CSV inspection reports.

<table>
  <tr>
    <td align="center">
      <img src="Images/template.jpg" width="300" />
      <p>Template Image</p>
    </td>
    <td align="center">
      <img src="Images/test.jpg" width="300" />
      <p>Test Image</p>
    </td>
  </tr>
</table>

## 🛠 Problem Statement
- Manual PCB inspection is slow, inconsistent, and expensive.
- Around 15–20% of PCBs contain manufacturing defects.
- Human fatigue reduces inspection accuracy.
- Each defective board can cost $50–500, increasing overall manufacturing loss.

## 💡 Proposed Solution
- AI-powered PCB defect detection using Deep Learning + Computer Vision.
- Real-time defect detection through a web-based interface.
- Consistent and reliable automated analysis.
- Export of reports (CSV, images, and logs).
- High accuracy using an EfficientNet-B4 model.

<table>
<td align="center">
      <img src="Images/Annotated Result.jpg" width="400" />
      <p>Annotated Result</p>
    </td>
</table>

## 🚀 Features

- **Automated Defect Detection**: Identifies 6 types of PCB defects (open, short, mousebite, spur, pinhole, spurious copper)
- **Real-time Processing**: Web interface for instant defect analysis
- **High Accuracy**: 98.34% test accuracy with EfficientNet-B4 model
- **Export Capabilities**: Download annotated images and CSV prediction logs
- **Interactive UI**: User-friendly web interface with parameter controls

---

## 📁 Project Structure

```
PCBDEFECT_DETECTION/
├── data/
├── src/
├── templates/
├── web_app.py                # FastAPI web application
├── requirements.txt
├── dataset/
├── training_outputs/
├── evaluation_outputs/
└── preprocess_example/
```

---

## 🏗️ System Architecture

<table>
  <tr>
    <td align="center">
      <img src="Images/System Architecture.png" width="1000" />
      <p>System Architecture</p>
    </td>
  </tr>
</table>

### 🧭 Overview
The system consists of three main components:
1. **Preprocessing Pipeline**: Image processing and ROI extraction
2. **Deep Learning Model**: EfficientNet-B4 classifier for defect classification
3. **Web Application**: FastAPI-based interface for real-time inference

### 💻 Technology Stack
- **Backend**: Python, FastAPI, OpenCV, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: PyTorch, Torchvision
- **Computer Vision**: OpenCV, NumPy
- **Visualization**: Matplotlib, scikit-learn

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
```bash
git clone <repository-url>
cd PCBDEFECT_DETECTION
```

```bash
python -m venv venv
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python src\build_dataset.py --data-root data --out-root dataset
```

```bash
python src\train_efficientnet_b4.py --data dataset --out training_outputs --epochs 20
```

```bash
uvicorn web_app:app --reload
```

---

## 🎯 Usage

### 🌐 Web Interface
1. Open:
```
http://localhost:8000
```

2. Upload template and test images  
3. Adjust parameters  
4. Run detection  
5. Download results (images, CSV, logs)

---

### 💻 Command Line
```bash
python src\preprocessing.py -t template.jpg -s test.jpg -o output
python src\train_efficientnet_b4.py --data dataset --epochs 20
python src\evaluate_model.py --data dataset --model training_outputs\model_best.pth
```

---

## 📊 Model Performance

- **Test Accuracy**: 98.34%

### Per-class Performance:
- mousebite: 97.64%
- open: 96.92%
- pinhole: 100.00%
- short: 99.56%
- spur: 97.96%
- spurious copper: 98.67%

---

## 📈 Outputs

### Training Outputs
- model_best.pth  
- loss_curve.jpg  
- accuracy_curve.jpg  
- confusion_matrix.jpg  

### Web Outputs
- Annotated images
- CSV logs
- Prediction reports
- Processing metrics

---

## ⚙️ PCB Report File
https://1drv.ms/b/c/c76b039bc7fe048f/EWffnzlXd5NAsjYGdhdAt80BBWRH-8QjBns4HNNX5lenrQ?e=T2E0Er

---

## 🐛 Troubleshooting
- Model not loading → check training_outputs
- CUDA issues → install correct PyTorch version
- Memory issues → reduce batch size
- Import issues → reinstall dependencies

---

## 🚀 Performance Tips
- Use GPU for faster inference
- Optimize image size for speed
- Tune thresholds for accuracy

---

## 📝 License
Educational and research purposes only.

---

## 🤝 Contributing
1. Fork repo  
2. Create branch  
3. Commit changes  
4. Pull request  

---

## 📞 Support
Raise an issue in repository.
