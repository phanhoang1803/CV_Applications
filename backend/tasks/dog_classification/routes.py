from flask import Blueprint, request, jsonify
from . import DogClassification

dog_classification_bp = Blueprint('dog_classification', __name__)

@dog_classification_bp.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("\nPOST: Dog classification")
        
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Preprocess the image file
        image_data = DogClassification.preprocess_image(file)
        
        # Make prediction
        prediction = DogClassification.predict(image_data)
        return jsonify({'prediction': prediction})
        
    return None
