from flask import Blueprint, request, jsonify
import os
import cv2
import base64
# from .CarDetection import CarDetection
from . import YOLO

car_detection_bp = Blueprint('car_detection', __name__)

@car_detection_bp.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            print("\nPOST: Car detection")
            
            # Check if the POST request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})
            
            file = request.files['file']
            
            # If the user does not select a file, the browser submits an empty file without a filename
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            temp_file_path = 'temp_image.jpg'
            file.save(temp_file_path)
            
            # Run YOLO model prediction on the uploaded image
            model = YOLO()
            _, _, _, res = model.predict(temp_file_path)
            
            print("Done predicting!")
            
            # Remove temporary file
            os.remove(temp_file_path)
            
            _, buffer = cv2.imencode('.jpg', res)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            print("Done econding!")
            
            # Return the detection results
            return jsonify({'prediction': encoded_image})
    
    except Exception as e:
        print("Error", e)
        pass
    
    return None
