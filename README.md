

# 🩻 Mammogram Image Analysis Web App

This project is a web-based application that uses a YOLOv5 deep learning model to analyze mammogram images for potential abnormalities. The app provides both visual insights and a downloadable PDF report of the findings to aid medical professionals in diagnostics.

---

## 🚀 Features

- 🔍 **YOLOv5-based Detection**: Utilizes a pretrained YOLOv5 model to detect abnormalities in mammogram images.
- 🖼️ **Side-by-side Display**: Original and processed image views for easy comparison.
- 📄 **PDF Report Generation**: Downloadable report containing diagnostic results.
- 🌐 **User-Friendly Interface**: Built with Flask and styled using Bootstrap for a clean and responsive layout.
- ⚠️ **Visual Risk Indicators**: BIRADS classification labels displayed with alert levels (e.g., `birads2`, `birads5`).

---

## 🗂️ Project Structure

```
├── app.py                # Flask backend for image processing and routing
├── best.pt               # YOLOv5 trained model for mammogram detection
├── pdf_generator.py      # Logic for generating the analysis report PDF
├── templates/
│   ├── index.html        # Upload interface
│   └── result.html       # Displays analysis results and download option
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and setup instructions
```

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mammogram-analysis-app.git
   cd mammogram-analysis-app
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv5**
   This app uses a custom-trained YOLOv5 model (`best.pt`). Make sure `yolov5` is cloned and added to your project directory:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   ```

---

## ▶️ Running the App

```bash
python app.py
```

Open your browser and visit:  
`http://127.0.0.1:5000`

---

## 📁 Upload & Analyze

1. Upload a mammogram image (JPG/PNG).
2. View results with BIRADS classifications.
3. Download the analysis report as a PDF.

---

## 🧠 Model Information

- **Model**: YOLOv5
- **Trained on**: Custom dataset of annotated mammogram images
- **Classes**: `birads0`, `birads2`, `birads5`, etc.
- **Format**: Torch `.pt` model (`best.pt`)

---

## 📄 Report Sample

The generated PDF includes:
- Original and processed images
- Detected findings
- Explanations per BIRADS label

---

## ⚠️ Disclaimer

> This application provides AI-assisted analysis **only**. The results are **not** a substitute for professional medical advice. Always consult a certified radiologist or healthcare provider for medical decisions.

---

## 🛠️ Tech Stack

- **Frontend**: HTML, Bootstrap
- **Backend**: Python, Flask
- **Model**: YOLOv5 (PyTorch)
- **PDF Generation**: ReportLab / FPDF (via `pdf_generator.py`)


