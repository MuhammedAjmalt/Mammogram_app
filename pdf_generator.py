import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(original_img_path, processed_img_path, detected_classes):
    try:
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"mammogram_report_{timestamp}.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Add title
        title = Paragraph("Mammogram Analysis Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Add images
        orig_img = Image(original_img_path, width=250, height=250)
        proc_img = Image(processed_img_path, width=250, height=250)
        
        # Create a table for images
        data = [[orig_img, proc_img]]
        table = Table(data)
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Add detection results
        results_text = []
        results_text.append(Paragraph("Detection Results:", styles['Heading2']))
        
        for detection in detected_classes:
            if isinstance(detection, dict):
                label = detection.get('label', 'Unknown')
                desc = detection.get('description', 'No description available')
                count = detection.get('count', 0)
                total = detection.get('total', 0)
                
                result_str = f"Class: {label}<br/>Description: {desc}<br/>Count: {count}/{total}<br/>"
                results_text.append(Paragraph(result_str, styles['Normal']))
                results_text.append(Spacer(1, 6))
            else:
                results_text.append(Paragraph(f"Detection: {str(detection)}", styles['Normal']))
                results_text.append(Spacer(1, 6))

        elements.extend(results_text)
        
        # Build PDF
        doc.build(elements)
        return pdf_path
        
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")
