# challan_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os
import time

def generate_challan(
    output_dir,
    timestamp,
    camera_id,
    vehicle_id,
    speed_kmh,
    plate_number,
    vehicle_image_path,
    plate_image_path
):
    """
    Generates a PDF challan for an overspeeding vehicle.
    """
    try:
        challan_filename = f"challan_{camera_id}_{plate_number}_{timestamp}.pdf"
        challan_path = os.path.join(output_dir, challan_filename)
        
        doc = SimpleDocTemplate(challan_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_text = f"E-Challan for Overspeeding Violation"
        story.append(Paragraph(f"<font size=16><b>{title_text}</b></font>", styles['Title']))
        story.append(Spacer(1, 12))

        # Violation Details
        details = [
            f"<b>Violation Date:</b> {timestamp}",
            f"<b>Camera ID:</b> {camera_id}",
            f"<b>Vehicle ID:</b> {vehicle_id}",
            f"<b>Detected Speed:</b> {speed_kmh} km/h",
            f"<b>License Plate:</b> {plate_number}",
        ]
        for detail in details:
            story.append(Paragraph(detail, styles['Normal']))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 24))

        # Evidence Images
        story.append(Paragraph("<b>Evidence:</b>", styles['Heading2']))
        story.append(Spacer(1, 12))

        if os.path.exists(vehicle_image_path):
            img_vehicle = Image(vehicle_image_path, width=400, height=225)
            story.append(img_vehicle)
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Captured Vehicle Image</b>", styles['Normal']))

        if os.path.exists(plate_image_path):
            img_plate = Image(plate_image_path, width=200, height=112)
            story.append(Spacer(1, 12))
            story.append(img_plate)
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Captured License Plate</b>", styles['Normal']))

        # Disclaimer
        story.append(Spacer(1, 36))
        story.append(Paragraph(
            "<font size=10><i>This document is for demonstration purposes only and does not constitute a legal fine or official notice.</i></font>",
            styles['Normal']
        ))

        doc.build(story)
        return challan_path
        
    except Exception as e:
        print(f"Error generating PDF challan: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing
    # Note: You need to have example images saved in the specified paths for this to work.
    sample_challan_path = generate_challan(
        output_dir="outputs/challans",
        timestamp="2025-09-15_22-30-00",
        camera_id="street_1",
        vehicle_id="12345",
        speed_kmh=120,
        plate_number="AB1234",
        vehicle_image_path="outputs/vehicles/vehicle_street_1_12345_2025-09-15_22-30-00.jpg",
        plate_image_path="outputs/number_plates/plate_street_1_AB1234_2025-09-15_22-30-00.jpg"
    )
    if sample_challan_path:
        print(f"Sample challan generated successfully at: {sample_challan_path}")