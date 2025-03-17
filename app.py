from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained YOLOv8 model


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

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
    
    # Define BIRADS descriptions
    birads_descriptions = {
        'birads0': 'Further imaging needed for a clearer diagnosis.',
        'birads2': 'Benign finding, no cause for concern.',
        'birads6': 'Strong indication of malignancy, follow-up required.',
        '0': 'No significant findings detected.'
    }
    
    # Count occurrences of each class
    class_counts = {}
    for cls in first_result.boxes.cls:
        class_num = int(cls)
        class_name = first_result.names[class_num]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Get the majority class
    if class_counts:
        majority_class_name = max(class_counts.items(), key=lambda x: x[1])[0]
        description = birads_descriptions.get(majority_class_name, 'Unknown classification')
        
        class_labels = [{
            'label': majority_class_name,  # Removed class ID prefix
            'description': description,
            'count': class_counts[majority_class_name],
            'total': len(first_result.boxes.cls)
        }]
    else:
        class_labels = []
    
    logger.info(f"Majority class with description: {class_labels}")
    
    # Save the processed results
    first_result.save(processed_filepath)
    
    return processed_filepath, class_labels  # Return both filepath and labels

if __name__ == '__main__':
    app.run(debug=True)