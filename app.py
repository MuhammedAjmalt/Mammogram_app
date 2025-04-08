from flask import Flask, render_template, request, send_file, session
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging
from pdf_generator import generate_pdf
import io
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained YOLOv8 model


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.secret_key = 'your_secret_key'  # Required for session

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check for preset image selection
    preset_image = request.form.get('preset_image')
    if preset_image:
        filepath = os.path.join('static', 'samples', preset_image)
        if os.path.exists(filepath):
            filename = preset_image
        else:
            return "Selected preset image not found"
    
    # If no preset image, handle uploaded file
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return "No image selected"
    
    processed_filepath, class_labels = process_image(filepath, filename)
    
    # Store paths and detected classes in session
    session['original_image_path'] = filepath
    session['processed_image_path'] = processed_filepath
    session['detected_classes'] = class_labels
    
    return render_template('result.html', 
                         original=filepath, 
                         processed=processed_filepath,
                         detected_classes=class_labels)

def process_image(filepath, filename):
    # Initialize YOLO model
    model = YOLO(r"D:\Downloads\mammo_detection\runs\detect\train19\weights\best.pt")
    
    # Process the uploaded image using filepath
    results = model(filepath)
    logger.info(f"Detection Results: {results}")
    
    # Generate processed image path
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    first_result = results[0]
    
    # Define BIRADS descriptions - remove birads5/6 completely
    birads_descriptions = {
        'birads0': 'Further imaging needed for a clearer diagnosis.',
        'birads2': 'Benign finding, no cause for concern.',
        '0': 'No significant findings detected.'
    }

    # Filter out birads5 and birads6 detections
    class_counts = {}
    filtered_results = []
    
    # Get allowed classes from descriptions
    allowed_classes = set(birads_descriptions.keys())
    
    for i, (box, cls, conf) in enumerate(zip(first_result.boxes.xyxy, first_result.boxes.cls, first_result.boxes.conf)):
        class_num = int(cls)
        class_name = first_result.names[class_num]
        
        # Only include classes that are in our allowed set
        if class_name in allowed_classes:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            filtered_results.append({
                'box': box.tolist(),
                'class': class_name,
                'conf': float(conf)
            })
    
    # Get the majority class and continue only if we have allowed detections
    if class_counts:
        majority_class_name = max(class_counts.items(), key=lambda x: x[1])[0]
        description = birads_descriptions.get(majority_class_name, 'Unknown classification')
        
        class_labels = [{
            'label': majority_class_name,
            'description': description,
            'count': class_counts[majority_class_name],
            'total': len(filtered_results)
        }]
        
        # Draw only allowed detections
        import cv2
        img = cv2.imread(filepath)
        for result in filtered_results:
            box = result['box']
            label = result['class']
            conf = result['conf']
            
            # Draw bounding box for allowed classes only
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label_text = f"{label} {conf:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the processed image
        cv2.imwrite(processed_filepath, img)
    else:
        class_labels = []
        # Save original image as processed if no allowed detections
        import shutil
        shutil.copy(filepath, processed_filepath)
    
    logger.info(f"Majority class with description: {class_labels}")
    return processed_filepath, class_labels

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        original_img_path = request.form.get('original_img')
        processed_img_path = request.form.get('processed_img')
        detected_classes_str = request.form.get('detected_classes', '[]')
        
        if isinstance(detected_classes_str, str):
            detected_classes = json.loads(detected_classes_str.replace("'", '"'))
        else:
            detected_classes = []
        
        pdf_path = generate_pdf(original_img_path, processed_img_path, detected_classes)
        filename = os.path.basename(pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name=filename)
    except Exception as e:
        app.logger.error(f"PDF generation error: {str(e)}")
        return f"Error generating PDF: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
