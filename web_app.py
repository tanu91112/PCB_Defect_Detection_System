
#!/usr/bin/env python3
"""
🎯 Goal
To build a Flask-based AI web application that detects defects in PCB (Printed Circuit Board) images using a deep learning model 
(EfficientNet-B4) and automatically generates a professional PDF inspection report with visual analytics and downloadable logs.

It’s a complete AI-powered PCB quality inspection system with automated reporting and visualization

"""

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response as FastAPIResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from fastapi.middleware.cors import CORSMiddleware

class SimpleCache:
    def __init__(self, default_timeout: int = 300):
        self._cache = {}
        self._default_timeout = default_timeout

    def get(self, key):
        value = self._cache.get(key)
        if value is None:
            return None
        expiry, item = value
        if expiry is not None and time.time() > expiry:
            self._cache.pop(key, None)
            return None
        return item

    def set(self, key, value, timeout: int = None):
        expiry = None
        if timeout is None:
            timeout = self._default_timeout
        if timeout > 0:
            expiry = time.time() + timeout
        self._cache[key] = (expiry, value)

import cv2
import numpy as np
import torch
import io
import os
import base64
import json
import time
import hashlib
from torch import nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

app = FastAPI()
os.makedirs("outputs", exist_ok=True)

# Configure caching with a single pure-Python cache instance
cache = SimpleCache(default_timeout=300)

# Templates
# Create a single Jinja2Templates instance and ensure a clean Jinja2 Environment
templates = Jinja2Templates(directory="templates")


# CORS (allow all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
model = None
classes = None


# ---------------- LOAD MODEL ----------------
def load_model():
    global model, classes
    try:
        model_path = "training_outputs/model_best.pth"
        classes_path = "training_outputs/classes.json"
        if not Path(model_path).exists() or not Path(classes_path).exists():
            print("Model or classes not found.")
            return False

        with open(classes_path, 'r') as f:
            classes = json.load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, len(classes))  # type: ignore
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


load_model()


# ---------------- HELPERS ----------------
def create_image_hash(image_data):
    """Create a hash of image data for caching"""
    return hashlib.md5(image_data).hexdigest()

def preprocess_image(image, img_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image).unsqueeze(0)


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"


def create_plot_base64_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def mask_from_diff(diff, thresh=30):
    _, th = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated

def align_affine_ecc(template_gray, test_gray, test_color):
    h, w = template_gray.shape
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cv2.findTransformECC(template_gray, test_gray, warp, cv2.MOTION_AFFINE, criteria)
        aligned_gray = cv2.warpAffine(test_gray, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        aligned_color = cv2.warpAffine(test_color, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned_gray, aligned_color
    except Exception:
        return test_gray, test_color


# ---------------- PDF GENERATION ----------------
def draw_page_border(cnv, doc):
    cnv.saveState()
    cnv.setStrokeColor(colors.black)  # black border per request
    cnv.setLineWidth(2)
    cnv.rect(20, 20, doc.pagesize[0] - 40, doc.pagesize[1] - 40)
    cnv.restoreState()


def generate_pdf_report(session_id, predictions, bar_plot_path=None, scatter_plot_path=None,
                        annotated_paths=None, pie_chart_path=None):


    """
    Professional report:
    - Two-column overview images (side-by-side)
    - Headings in blue, border black
    - Saves and returns the path to the generated PDF
    """
    file_path = f"outputs/report_{session_id}.pdf"
    os.makedirs("outputs", exist_ok=True)

    doc = SimpleDocTemplate(file_path, pagesize=letter,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
    name='ReportTitle',
    fontName='Helvetica-Bold',
    fontSize=22,
    alignment=1,
    textColor=colors.HexColor('#1A237E'),
    leading=28
))

    styles.add(ParagraphStyle(
    name='ReportSub',
    fontName='Helvetica',
    fontSize=11,
    alignment=1,
    textColor=colors.HexColor('#37474F'),
))

    styles.add(ParagraphStyle(
    name='HeadingBlue',
    fontName='Helvetica-Bold',
    fontSize=14,
    alignment=0,
    textColor=colors.HexColor('#0D47A1'),
    leading=18
))

    styles.add(ParagraphStyle(
    name='NormalText',
    fontName='Helvetica',
    fontSize=10,
    leading=14
))

    normal = styles['NormalText']


    elements = []

    # Cover: optional logo/banner
    logo_path = "outputs/logo.png"
    if os.path.exists(logo_path):
        try:
            elements.append(RLImage(logo_path, width=6.6 * inch, height=1.0 * inch))
            elements.append(Spacer(1, 8))
        except Exception:
            pass

    # Title & meta
    elements.append(Paragraph("<b>A Generated Report on PCB Defect Detection using AI</b>", styles['ReportTitle']))
    elements.append(Paragraph("By: Tanu Chandravanshi<br/>VIT Bhopal University<br/>Date: " +
                          datetime.now().strftime('%d-%m-%Y'), styles['ReportSub']))
   
    elements.append(Paragraph("AI-Powered Quality Inspection using EfficientNet-B4", styles['ReportSub']))
    elements.append(Spacer(1, 24))

    info_html = (
        f"<b>Report ID:</b> {session_id} &nbsp;&nbsp;&nbsp; "
        f"<b>Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        f"<b>Model Used:</b> EfficientNet-B4 &nbsp;&nbsp;&nbsp; "
        f"<b>Total Defects Detected:</b> {len(predictions)}"
    )
    elements.append(Paragraph(info_html, normal))
    elements.append(Spacer(1, 12))

    # Overview heading
    elements.append(Paragraph("<b>Overview Images</b>", styles['HeadingBlue']))
    elements.append(Spacer(1, 6))

    # Prepare overview images (only existing ones)
    overview_items = []
    raw_img_paths = [
        ("Template Image", f"outputs/template_{session_id}.png"),
        ("Test Image", f"outputs/test_{session_id}.png"),
        ("Difference Image", f"outputs/diff_{session_id}.png"),
        ("Binary Mask", f"outputs/mask_{session_id}.png"),
        ("Bar Plot - Defect Count", bar_plot_path),
        ("Scatter Plot - Defect Positions", scatter_plot_path),
        ("Pie Chart - Class Distribution", pie_chart_path),

    ]
    for title, p in raw_img_paths:
        if p and os.path.exists(p):
            # nested table for each cell: title + image (keeps title above)
            try:
                img = RLImage(p, width=3.0 * inch, height=2.0 * inch)  # compact size for side-by-side
                nested = Table([[Paragraph(f"<b>{title}</b>", styles['HeadingBlue'])],
                                [img]], colWidths=[3.0 * inch])
                nested.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]))
                overview_items.append(nested)
            except Exception:
                # fallback: skip problematic images
                pass

    # Build rows of two columns
    if overview_items:
        rows = []
        for i in range(0, len(overview_items), 2):
            left = overview_items[i]
            right = overview_items[i + 1] if i + 1 < len(overview_items) else ''
            rows.append([left, right])
        table_overview = Table(rows, colWidths=[3.15 * inch, 3.15 * inch], hAlign='CENTER')
        table_overview.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table_overview)


    # --- Detected defects summary (each defect as own table, green header kept) ---
    elements.append(Paragraph("<b>Detected Defects Summary</b>", styles['HeadingBlue']))
    elements.append(Spacer(1, 8))
    if predictions:
        # Table header
        defect_table_data = [
            ["Defect ID", "Class", "Confidence", "Position (x, y)", "Size (w×h)"]
        ]

        # Table rows for each defect
        for i, p in enumerate(predictions, start=1):
            bx, by, bw, bh = p['bbox']
            defect_table_data.append([
                str(i),
                p["class"],
                f"{p['confidence']:.2f}",
                f"({bx}, {by})",
                f"{bw} × {bh}"
            ])

        # Define column widths
        col_widths = [1.0 * inch, 1.6 * inch, 1.0 * inch, 1.7 * inch, 1.3 * inch]

        # Create one unified table
        defect_table = Table(defect_table_data, colWidths=col_widths, hAlign='CENTER')
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.darkgreen),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(defect_table)
        elements.append(Spacer(1, 12))
    else:
        elements.append(Paragraph("No defects detected.", normal))
    

   # Move directly to annotated results (no PageBreak)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Annotated Results</b>", styles['HeadingBlue']))
    elements.append(Spacer(1, 6))
    if annotated_paths:
        for img in annotated_paths:
            if os.path.exists(img):
                try:
                    elements.append(RLImage(img, width=6.8 * inch, height=4.8 * inch))
                    elements.append(Spacer(1, 8))
                except Exception:
                    pass
    else:
        elements.append(Paragraph("No annotated image available.", normal))

    # Summary & conclusion
    elements.append(Paragraph("<b>Summary</b>", styles['HeadingBlue']))
    elements.append(Spacer(1, 6))
    total = len(predictions)
    high_conf = sum(1 for p in predictions if p['confidence'] > 0.8)
    summary_text = (
        f"Out of <b>{total}</b> detected defects, <b>{high_conf}</b> were high-confidence "
        f"(confidence > 0.8). The inspection identified potential PCB defects (open circuits, shorts, "
        f"missing components)."
    )
    elements.append(Paragraph(summary_text, normal))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Conclusion</b>", styles['HeadingBlue']))
    elements.append(Spacer(1, 6))
    conclusion_text = (
        "This AI-powered PCB defect detection system (EfficientNet-B4) demonstrates strong performance "
        "for automated visual inspection. Integrating real-time detection and adaptive thresholding "
        "would be recommended for production deployments."
    )
    elements.append(Paragraph(conclusion_text, normal))

    # Build the PDF with black border
    doc.build(elements, onFirstPage=draw_page_border, onLaterPages=draw_page_border)
    return file_path


# ---------------- ROUTES ----------------
@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/detect')
async def detect_defects(template: UploadFile = File(...), test: UploadFile = File(...),
                         thresh: int = Form(30), min_area: int = Form(50),
                         conf_thresh: float = Form(0.6), align: str = Form('false')):
    try:
        start_time = time.time()
        if model is None:
            return JSONResponse({'success': False, 'error': 'Model not loaded'})

        use_alignment = str(align).lower() in ('1', 'true', 'yes')

        # Read image data for hashing
        template_data = await template.read()
        test_data = await test.read()
        
        # Create cache key based on image hashes and parameters
        template_hash = create_image_hash(template_data)
        test_hash = create_image_hash(test_data)
        cache_key = f"prediction_{template_hash}_{test_hash}_{thresh}_{min_area}_{conf_thresh}"
        
        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            print(f"🎯 Cache hit for key: {cache_key}")
            return JSONResponse(cached_result)

        # Decode images
        template_img = cv2.imdecode(np.frombuffer(template_data, np.uint8), cv2.IMREAD_COLOR)
        test_img = cv2.imdecode(np.frombuffer(test_data, np.uint8), cv2.IMREAD_COLOR)
        if template_img is None or test_img is None:
            return JSONResponse({'success': False, 'error': 'Invalid image format'})

        template_blur = cv2.GaussianBlur(template_img, (5, 5), 0)
        test_blur = cv2.GaussianBlur(test_img, (5, 5), 0)
        gray_t = cv2.cvtColor(template_blur, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.cvtColor(test_blur, cv2.COLOR_BGR2GRAY)
        if use_alignment:
            gray_s, test_img = align_affine_ecc(gray_t, gray_s, test_img)
            diff = cv2.absdiff(gray_s, gray_t)
        else:
            diff = cv2.absdiff(gray_s, gray_t)

        mask = mask_from_diff(diff, thresh=thresh)

        
        mask = cv2.bitwise_and(mask, mask)

        # contours -> rois
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area:
                rois.append((x, y, w, h))

        # Predictions on ROIs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = []
        for (x, y, w, h) in rois:
            crop = test_img[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            with torch.no_grad():
                image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(image_rgb)
                inp = preprocess_image(pil).to(device)
                outputs = model(inp)
                probs = torch.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, 1)
                pred_class = classes[idx.item()] if classes else str(idx.item())
                if conf.item() >= conf_thresh:
                    predictions.append({
                        'class': pred_class,
                        'confidence': float(conf.item()),
                        'bbox': (int(x), int(y), int(w), int(h))
                    })

        # Annotate test image
        annotated = test_img.copy()
        for p in predictions:
            x, y, w, h = p['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, f"{p['class']} ({p['confidence']:.2f})",
                        (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        session_id = str(int(time.time()))

        # Save intermediate images
        cv2.imwrite(f"outputs/template_{session_id}.png", template_img)
        cv2.imwrite(f"outputs/test_{session_id}.png", test_img)
        cv2.imwrite(f"outputs/diff_{session_id}.png", diff)
        cv2.imwrite(f"outputs/mask_{session_id}.png", mask)
        annotated_path = f"outputs/annotated_{session_id}.png"
        cv2.imwrite(annotated_path, annotated)

        # Save JSON results
        results_json = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "session_id": session_id,
            "predictions": predictions
        }
        json_path = f"outputs/results_{session_id}.json"
        with open(json_path, 'w') as jf:
            json.dump(results_json, jf, indent=2)

        # Class counts
        class_counts = {}
        for p in predictions:
            class_counts[p['class']] = class_counts.get(p['class'], 0) + 1

        # --- PIE CHART ---
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        if class_counts:
            labels = list(class_counts.keys())
            sizes = list(class_counts.values())
            colors_list = plt.cm.get_cmap('Paired')(np.linspace(0, 1, len(labels)))
            colors_list = list(colors_list)


            pie_result = ax3.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                startangle=140, colors=colors_list,
                textprops={'fontsize': 8}
            )

            wedges, texts = pie_result[0], pie_result[1]
            autotexts = pie_result[2] if len(pie_result) > 2 else []
            ax3.set_title("Defect Class Distribution (Pie Chart)")
            ax3.axis('equal')
        else:
            ax3.text(0.5, 0.5, "No defects detected", ha='center', va='center', fontsize=12)
            ax3.axis('off')

        pie_chart_path = f"outputs/pie_chart_{session_id}.png"
        fig3.savefig(pie_chart_path, bbox_inches='tight')
        pie_chart_b64 = create_plot_base64_from_fig(fig3)
        plt.close(fig3)

        # --- BAR CHART ---
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if class_counts:
            ax1.bar(list(class_counts.keys()), list(class_counts.values()), color='lightgreen')
            ax1.set_title("Defect Count per Class")
            ax1.set_ylabel("Count")
        else:
            ax1.text(0.5, 0.5, "No defects detected", ha='center', va='center', fontsize=12)
            ax1.axis('off')
        bar_plot_path = f"outputs/bar_plot_{session_id}.png"
        fig1.savefig(bar_plot_path, bbox_inches='tight')
        bar_plot_b64 = create_plot_base64_from_fig(fig1)
        plt.close(fig1)

        # --- SCATTER PLOT ---
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if predictions:
            xs, ys, labels = [], [], []
            for p in predictions:
                x, y, w, h = p['bbox']
                xs.append(x + w / 2)
                ys.append(y + h / 2)
                labels.append(p['class'])
            ax2.scatter(xs, ys, c='green')
            for i, lbl in enumerate(labels):
                ax2.annotate(lbl, (xs[i], ys[i]), textcoords="offset points", xytext=(3, 3), fontsize=8)
            ax2.set_title("Defect Position Scatter Plot")
            ax2.set_xlabel("X Position")
            ax2.set_ylabel("Y Position")
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "No defects detected", ha='center', va='center', fontsize=12)
            ax2.axis('off')
        scatter_plot_path = f"outputs/scatter_plot_{session_id}.png"
        fig2.savefig(scatter_plot_path, bbox_inches='tight')
        scatter_plot_b64 = create_plot_base64_from_fig(fig2)
        plt.close(fig2)

        # Generate PDF
        report_path = generate_pdf_report(
            session_id, predictions,
            bar_plot_path, scatter_plot_path,
            [annotated_path],
            pie_chart_path=pie_chart_path
        )

        processing_time = time.time() - start_time
        distribution = {k: int(v) for k, v in class_counts.items()}

        # Calculate overall accuracy based on confidence scores
        overall_accuracy = 0.0
        if predictions:
            confidences = [p['confidence'] for p in predictions]
            overall_accuracy = sum(confidences) / len(confidences) * 100

        # Prepare response with image URLs for individual download
        response_data = {
        'success': True,
        'session_id': session_id,
        'processing_time': processing_time,
        'images': {
            'template': image_to_base64(template_img),
            'test': image_to_base64(test_img),
            'diff': image_to_base64(diff),
            'mask': image_to_base64(mask),
            'annotated': image_to_base64(annotated),
        },
        'image_urls': {
            'template_url': f'/download_image/{session_id}/template',
            'test_url': f'/download_image/{session_id}/test', 
            'diff_url': f'/download_image/{session_id}/diff',
            'mask_url': f'/download_image/{session_id}/mask',
            'annotated_url': f'/download_image/{session_id}/annotated'
        },
        'plot_urls': {
            'bar_url': f'/download_plot/{session_id}/bar',
            'scatter_url': f'/download_plot/{session_id}/scatter',
            'pie_url': f'/download_plot/{session_id}/pie'
        },
        'bar_plot': bar_plot_b64,
        'scatter_plot': scatter_plot_b64,
        'pie_chart': pie_chart_b64,
        'defect_distribution': distribution,
        'predictions': predictions,
        'defects_found': len(predictions),
        'high_confidence': sum(1 for p in predictions if p['confidence'] > 0.8),
        'accuracy': overall_accuracy
        }
        
        # Cache the result for 5 minutes
        cache.set(cache_key, response_data)
        print(f"💾 Cached result with key: {cache_key}")
        
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)})


# ------------------------ DOWNLOAD ROUTES ------------------------

@app.get('/download_report/{session_id}')
async def download_report(session_id: str):
    path = f"outputs/report_{session_id}.pdf"
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Report not found"}, status_code=404)
    return FileResponse(path, media_type='application/pdf', filename=os.path.basename(path))


@app.get('/download_image/{session_id}')
async def download_image(session_id: str):
    """Download annotated image (backward compatibility)"""
    path = f"outputs/annotated_{session_id}.png"
    if os.path.exists(path):
        return FileResponse(path, media_type='image/png', filename=f"annotated_image_{session_id}.png")
    return JSONResponse(content={"error": "Image not found"}, status_code=404)

@app.get('/download_image/{session_id}/{image_type}')
async def download_specific_image(session_id: str, image_type: str):
    """Download specific image type with proper filename"""
    filename_map = {
        'template': f'template_{session_id}.png',
        'test': f'test_{session_id}.png', 
        'diff': f'diff_{session_id}.png',
        'mask': f'mask_{session_id}.png',
        'annotated': f'annotated_{session_id}.png'
    }
    
    # Friendly .jpg filenames per requirement
    download_name_map = {
        'template': 'original_image.jpg',
        'test': 'processed_image.jpg',
        'diff': 'difference_image.jpg', 
        'mask': 'mask_image.jpg',
        'annotated': 'predicted_image.jpg'
    }
    
    if image_type not in filename_map:
        return JSONResponse(content={"error": "Invalid image type"}, status_code=404)
        
    path = f"outputs/{filename_map[image_type]}"
    if os.path.exists(path):
        # Guess media type by extension
        media_type = 'image/png' if path.lower().endswith('.png') else 'application/octet-stream'
        return FileResponse(path, media_type=media_type, filename=download_name_map[image_type])
    return JSONResponse(content={"error": "Image not found"}, status_code=404)


@app.get('/download_plot/{session_id}/{plot_type}')
async def download_plot(session_id: str, plot_type: str):
    """Download saved plots (bar, scatter, pie) with friendly filenames"""
    plot_filename_map = {
        'bar': f'bar_plot_{session_id}.png',
        'scatter': f'scatter_plot_{session_id}.png',
        'pie': f'pie_chart_{session_id}.png'
    }

    plot_download_name_map = {
        'bar': 'defect_count_bar_plot.jpg',
        'scatter': 'defect_scatter_plot.jpg',
        'pie': 'defect_distribution_pie_chart.jpg'
    }

    if plot_type not in plot_filename_map:
        return JSONResponse(content={"error": "Invalid plot type"}, status_code=404)

    path = f"outputs/{plot_filename_map[plot_type]}"
    if os.path.exists(path):
        media_type = 'image/png' if path.lower().endswith('.png') else 'application/octet-stream'
        return FileResponse(path, media_type=media_type, filename=plot_download_name_map[plot_type])
    return JSONResponse(content={"error": "Plot not found"}, status_code=404)


@app.get('/download_log/{session_id}')
async def download_log(session_id: str):
    json_path = f"outputs/results_{session_id}.json"
    if not os.path.exists(json_path):
        return JSONResponse(content={"error": "Log not found"}, status_code=404)

    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except Exception:
        return JSONResponse(content={"error": "Error reading log"}, status_code=500)

    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Session ID', 'Defect ID', 'Class', 'Confidence', 'Bounding Box (x,y,w,h)'])
    for i, d in enumerate(results.get('predictions', [])):
        writer.writerow([
            results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            session_id, i + 1,
            d.get('class', ''),
            f"{d.get('confidence', 0):.4f}",
            str(d.get('bbox', ''))
        ])
    csv_data = output.getvalue()
    output.close()
    headers = {
        "Content-Disposition": f"attachment; filename=prediction_log_{session_id}.csv",
        "Content-Type": "text/csv"
    }
    return FastAPIResponse(content=csv_data, headers=headers)


#uvicorn web_app:app --reload
